"""
고칼륨혈증 감지 모델 학습

ECG 파형에서 고칼륨혈증(K>=5.5)을 감지하는 이진 분류 모델
ResNet-34 from scratch (Tarabanis 방법론 동일)
"""
import argparse
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tqdm import tqdm
import h5py

from resnet34_ecg import SEResNet34ECGWithTabular


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al. 2017) — hard example에 집중, easy example 가중치 감소"""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # class별 가중치 (tensor)
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)  # 정답 클래스의 확률
        focal_weight = (1 - pt) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_weight = alpha_t * focal_weight
        return (focal_weight * ce).mean()


class HyperkalemiaDataset(Dataset):
    def __init__(self, ecg_h5_path, hk_h5_path, indices=None, lead_indices=None):
        self.ecg_h5_path = ecg_h5_path
        self._ecg_h5 = None
        self.lead_indices = lead_indices
        self.add_lead3 = False

        with h5py.File(hk_h5_path, "r") as f:
            all_ecg_indices = f["indices"][:]
            all_labels = f["hk_label"][:]
            all_sids = f["subject_id"][:]

        if indices is not None:
            self.ecg_indices = all_ecg_indices[indices]
            self.labels = torch.tensor(all_labels[indices], dtype=torch.long)
            self.subject_ids = all_sids[indices]
        else:
            self.ecg_indices = all_ecg_indices
            self.labels = torch.tensor(all_labels, dtype=torch.long)
            self.subject_ids = all_sids

        with h5py.File(ecg_h5_path, "r") as f:
            sorted_order = np.argsort(self.ecg_indices)
            sorted_idx = self.ecg_indices[sorted_order]
            raw_numeric = f["numeric_features"][sorted_idx]
            raw_patient = f["patient_features"][sorted_idx]
            reverse_order = np.argsort(sorted_order)
            numeric = raw_numeric[reverse_order].astype(np.float32)
            patient = raw_patient[reverse_order].astype(np.float32)
            np.nan_to_num(numeric, copy=False, nan=0.0)
            np.nan_to_num(patient, copy=False, nan=0.0)
            self.numeric = torch.from_numpy(numeric)
            self.patient = torch.from_numpy(patient)

    @property
    def ecg_h5(self):
        if self._ecg_h5 is None:
            self._ecg_h5 = h5py.File(self.ecg_h5_path, "r", rdcc_nbytes=64 * 1024 * 1024)
        return self._ecg_h5

    def __len__(self):
        return len(self.ecg_indices)

    def __getitem__(self, idx):
        real_idx = int(self.ecg_indices[idx])
        waveform = torch.tensor(self.ecg_h5["waveform"][real_idx], dtype=torch.float32)
        if self.lead_indices is not None:
            waveform = waveform[:, self.lead_indices]
        if self.add_lead3:
            lead3 = waveform[:, 1:2] - waveform[:, 0:1]
            waveform = torch.cat([waveform, lead3], dim=1)
        return waveform, self.numeric[idx], self.patient[idx], self.labels[idx]

    def __del__(self):
        if self._ecg_h5 is not None:
            self._ecg_h5.close()


def split_by_patient(hk_h5_path, train_ratio=0.7, val_ratio=0.1, seed=42,
                     eval_max_time_diff=None):
    """Patient-level split. eval_max_time_diff: val/test에만 적용할 시간 필터 (초)"""
    rng = np.random.RandomState(seed)
    with h5py.File(hk_h5_path, "r") as f:
        subject_ids = f["subject_id"][:]
        time_diffs = f["time_diff_sec"][:] if "time_diff_sec" in f else None
    unique = np.unique(subject_ids)
    rng.shuffle(unique)
    n = len(unique)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_p = set(unique[:n_train])
    val_p = set(unique[n_train:n_train + n_val])
    train_idx = np.where(np.isin(subject_ids, list(train_p)))[0]
    val_idx = np.where(np.isin(subject_ids, list(val_p)))[0]
    test_idx = np.where(np.isin(subject_ids, list(set(unique[n_train + n_val:]))))[0]

    # 비대칭 윈도우: val/test는 좁은 시간 범위만 사용
    if eval_max_time_diff is not None and time_diffs is not None:
        val_mask = time_diffs[val_idx] <= eval_max_time_diff
        test_mask = time_diffs[test_idx] <= eval_max_time_diff
        print(f"  Eval time filter ({eval_max_time_diff/3600:.0f}h): "
              f"val {val_mask.sum():,}/{len(val_idx):,}, "
              f"test {test_mask.sum():,}/{len(test_idx):,}")
        val_idx = val_idx[val_mask]
        test_idx = test_idx[test_mask]

    return train_idx, val_idx, test_idx


def compute_class_weights(hk_h5_path, indices, n_classes=2):
    with h5py.File(hk_h5_path, "r") as f:
        labels = f["hk_label"][:]
    labels = labels[indices]
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    weights = len(labels) / (n_classes * counts)
    weights[counts == 0] = 1.0
    return torch.tensor(weights, dtype=torch.float32)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4, save_path="best_model.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.best_loss = float("inf")
        self.counter = 0
        self.best_epoch = 0

    def step(self, val_loss, model, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.save_path)
            return False
        self.counter += 1
        return self.counter >= self.patience


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc=f"  Epoch {epoch} [Train]", leave=False,
                bar_format="{l_bar}{bar:30}{r_bar}")
    for wf, num, pat, lab in pbar:
        wf, num, pat, lab = wf.to(device), num.to(device), pat.to(device), lab.to(device)
        optimizer.zero_grad()
        logits = model(wf, num, pat)
        loss = criterion(logits, lab)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * lab.size(0)
        correct += (logits.argmax(1) == lab).sum().item()
        total += lab.size(0)
        pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.1%}")
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="Val"):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_probs, all_labels = [], [], []
    pbar = tqdm(loader, desc=f"         [{desc}]", leave=False,
                bar_format="{l_bar}{bar:30}{r_bar}")
    for wf, num, pat, lab in pbar:
        wf, num, pat, lab = wf.to(device), num.to(device), pat.to(device), lab.to(device)
        logits = model(wf, num, pat)
        loss = criterion(logits, lab)
        total_loss += loss.item() * lab.size(0)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(1)
        correct += (preds == lab).sum().item()
        total += lab.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())
        all_labels.extend(lab.cpu().numpy())
        pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.1%}")
    return (total_loss / total, correct / total,
            np.array(all_preds), np.array(all_probs), np.array(all_labels))


def main():
    parser = argparse.ArgumentParser(description="Hyperkalemia Detection (ResNet-34)")
    parser.add_argument("--ecg-data", default="g:/AIEKG/ml/data/ecg_preprocessed.h5")
    parser.add_argument("--hk-data", default="g:/AIEKG/ml/data/ecg_hyperkalemia_v2.h5")
    parser.add_argument("--output-dir", default="g:/AIEKG/ml/checkpoints/hyperkalemia-se-resnet34")
    parser.add_argument("--eval-window-hours", type=float, default=2,
                        help="Val/Test matching window (hours). Train uses full dataset.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--lr-factor", type=float, default=0.8)
    parser.add_argument("--es-patience", type=int, default=7)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\n[1/4] Data split (7:1:2)...")
    eval_max_sec = args.eval_window_hours * 3600
    train_idx, val_idx, test_idx = split_by_patient(
        args.hk_data, train_ratio=0.7, val_ratio=0.1,
        eval_max_time_diff=eval_max_sec
    )
    lead_indices = [0, 1]

    train_ds = HyperkalemiaDataset(args.ecg_data, args.hk_data, train_idx, lead_indices)
    val_ds = HyperkalemiaDataset(args.ecg_data, args.hk_data, val_idx, lead_indices)
    test_ds = HyperkalemiaDataset(args.ecg_data, args.hk_data, test_idx, lead_indices)
    for ds in [train_ds, val_ds, test_ds]:
        ds.add_lead3 = True

    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")
    train_labels = train_ds.labels.numpy()
    print(f"  Train pos: {train_labels.sum():,} / neg: {(train_labels==0).sum():,}")

    # WeightedRandomSampler: 양성/음성 균등 샘플링 (1:12.7 불균형 해결)
    sample_weights = np.where(train_labels == 1, 1.0 / train_labels.sum(),
                              1.0 / (len(train_labels) - train_labels.sum()))
    sample_weights = torch.tensor(sample_weights, dtype=torch.float64)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             pin_memory=True, num_workers=0)

    print("\n[2/4] SE-ResNet-34 (from scratch, Kwon 2024)...")
    n_numeric = train_ds.numeric.shape[1]
    n_patient = train_ds.patient.shape[1]
    model = SEResNet34ECGWithTabular(
        n_leads=3, n_numeric=n_numeric, n_patient=n_patient,
        n_classes=2, dropout=args.dropout
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience
    )
    early_stopping = EarlyStopping(
        patience=args.es_patience, save_path=str(output_dir / "best_model.pt")
    )

    start_epoch = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_auroc": []}
    ckpt_path = output_dir / "checkpoint.pt"
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        history = ckpt["history"]
        early_stopping.best_loss = ckpt["best_val_loss"]
        early_stopping.best_epoch = ckpt["best_epoch"]
        early_stopping.counter = ckpt["es_counter"]
        start_epoch = ckpt["epoch"] + 1
        print(f"  Resumed from epoch {start_epoch}")

    print(f"\n[3/4] Training ({args.epochs} epochs, Adam lr={args.lr})...")
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc, _, val_probs, val_labels = evaluate(
            model, val_loader, criterion, device, desc="Val"
        )
        val_auroc = roc_auc_score(val_labels, val_probs) if len(np.unique(val_labels)) > 1 else 0.0
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auroc"].append(val_auroc)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:2d} | "
              f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.3f} auroc={val_auroc:.4f} | "
              f"lr={lr:.1e} | {elapsed:.0f}s")

        torch.save({
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "history": history, "best_val_loss": early_stopping.best_loss,
            "best_epoch": early_stopping.best_epoch, "es_counter": early_stopping.counter,
        }, str(ckpt_path))

        if early_stopping.step(val_loss, model, epoch):
            print(f"  Early stopping at epoch {epoch} (best: {early_stopping.best_epoch})")
            break

    print(f"\n[4/4] Test evaluation (best epoch: {early_stopping.best_epoch})...")
    model.load_state_dict(torch.load(str(output_dir / "best_model.pt"), weights_only=True))
    test_loss, test_acc, test_preds, test_probs, test_labels = evaluate(
        model, test_loader, criterion, device, desc="Test"
    )
    test_auroc = roc_auc_score(test_labels, test_probs)

    print(f"\n  Test AUROC: {test_auroc:.4f}")
    print(f"  Test Accuracy: {test_acc:.1%}")
    print()
    print(classification_report(test_labels, test_preds,
                                target_names=["Normal K", "Hyperkalemia"]))
    print("Confusion Matrix:")
    print(confusion_matrix(test_labels, test_preds))

    results = {
        "test_auroc": float(test_auroc), "test_acc": float(test_acc),
        "test_loss": float(test_loss), "best_epoch": early_stopping.best_epoch,
        "args": vars(args), "history": {k: [float(v) for v in vs] for k, vs in history.items()},
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
