"""
발작성 AF 감지 모델 학습 — Tarabanis et al. (2025) 방법론

ResNet-34 from scratch, Adam, categorical CE,
LR scheduler (patience 2, factor 0.8), early stopping (patience 5), 100 epochs
"""
import argparse
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tqdm import tqdm
import h5py

from resnet34_ecg import ResNet34ECGWithTabular


# ============================================================
# Dataset (기존과 동일)
# ============================================================
class ParoxysmalAFDataset(Dataset):
    def __init__(self, ecg_h5_path, paf_h5_path, indices=None, lead_indices=None,
                 use_clinical=False):
        self.ecg_h5_path = ecg_h5_path
        self._ecg_h5 = None
        self.lead_indices = lead_indices
        self.add_lead3 = False

        with h5py.File(paf_h5_path, "r") as f:
            all_ecg_indices = f["indices"][:]
            all_labels = f["paf_label"][:]
            all_sids = f["subject_id"][:]
            all_clinical = f["clinical_features"][:] if use_clinical and "clinical_features" in f else None

        if indices is not None:
            self.ecg_indices = all_ecg_indices[indices]
            self.labels = torch.tensor(all_labels[indices], dtype=torch.long)
            self.subject_ids = all_sids[indices]
            if all_clinical is not None:
                all_clinical = all_clinical[indices]
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
            # clinical features (DM, HF, MI, AHT) 결합
            if all_clinical is not None:
                clinical = all_clinical.astype(np.float32)
                numeric = np.concatenate([numeric, clinical], axis=1)
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


def split_by_patient(paf_h5_path, train_ratio=0.7, val_ratio=0.1, seed=42):
    """Tarabanis: 7:1:2 split"""
    rng = np.random.RandomState(seed)
    with h5py.File(paf_h5_path, "r") as f:
        subject_ids = f["subject_id"][:]
    unique = np.unique(subject_ids)
    rng.shuffle(unique)
    n = len(unique)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_p = set(unique[:n_train])
    val_p = set(unique[n_train:n_train + n_val])
    test_p = set(unique[n_train + n_val:])
    train_idx = np.where(np.isin(subject_ids, list(train_p)))[0]
    val_idx = np.where(np.isin(subject_ids, list(val_p)))[0]
    test_idx = np.where(np.isin(subject_ids, list(test_p)))[0]
    return train_idx, val_idx, test_idx


def compute_class_weights(paf_h5_path, indices, n_classes=2):
    with h5py.File(paf_h5_path, "r") as f:
        labels = f["paf_label"][:]
    labels = labels[indices]
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    weights = len(labels) / (n_classes * counts)
    weights[counts == 0] = 1.0
    return torch.tensor(weights, dtype=torch.float32)


# ============================================================
# Train / Eval
# ============================================================
class EarlyStopping:
    """Tarabanis: patience 5, monitor val loss"""
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


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Paroxysmal AF Detection (ResNet-34)")
    parser.add_argument("--ecg-data", default="g:/AIEKG/ml/data/ecg_preprocessed.h5")
    parser.add_argument("--paf-data", default="g:/AIEKG/ml/data/ecg_paroxysmal_af.h5")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)       # Tarabanis: 100
    parser.add_argument("--lr", type=float, default=1e-3)        # Adam default
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr-patience", type=int, default=2)    # Tarabanis: 2
    parser.add_argument("--lr-factor", type=float, default=0.8)  # Tarabanis: 0.8
    parser.add_argument("--es-patience", type=int, default=5)    # Tarabanis: 5
    parser.add_argument("--use-clinical", action="store_true",
                        help="Add clinical features (DM, HF, MI, AHT)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.output_dir is None:
        suffix = "paroxysmal-af-resnet34"
        if args.use_clinical:
            suffix += "-clinical"
        args.output_dir = f"g:/AIEKG/ml/checkpoints/{suffix}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 데이터 (7:1:2 split, Tarabanis 방식)
    print("\n[1/4] Data split (7:1:2)...")
    train_idx, val_idx, test_idx = split_by_patient(args.paf_data, train_ratio=0.7, val_ratio=0.1)
    lead_indices = [0, 1]  # Lead I, II

    train_ds = ParoxysmalAFDataset(args.ecg_data, args.paf_data, train_idx, lead_indices,
                                   use_clinical=args.use_clinical)
    val_ds = ParoxysmalAFDataset(args.ecg_data, args.paf_data, val_idx, lead_indices,
                                 use_clinical=args.use_clinical)
    test_ds = ParoxysmalAFDataset(args.ecg_data, args.paf_data, test_idx, lead_indices,
                                  use_clinical=args.use_clinical)
    for ds in [train_ds, val_ds, test_ds]:
        ds.add_lead3 = True  # Lead III = II - I

    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")
    train_labels = train_ds.labels.numpy()
    print(f"  Train pos: {train_labels.sum():,} / neg: {(train_labels==0).sum():,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             pin_memory=True, num_workers=0)

    class_weights = compute_class_weights(args.paf_data, train_idx)
    print(f"  Class weights: {class_weights.tolist()}")

    # 모델: ResNet-34 from scratch (Tarabanis 방법론)
    print("\n[2/4] ResNet-34 (from scratch, Tarabanis method)...")
    n_numeric = train_ds.numeric.shape[1]
    n_patient = train_ds.patient.shape[1]

    model = ResNet34ECGWithTabular(
        n_leads=3, n_numeric=n_numeric, n_patient=n_patient,
        n_classes=2, dropout=args.dropout
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_params:,} (all trainable: {n_trainable:,})")

    # Tarabanis: categorical CE + class weighting
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Tarabanis: Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Tarabanis: LR scheduler (patience 2, factor 0.8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience
    )

    # Tarabanis: early stopping (patience 5)
    early_stopping = EarlyStopping(
        patience=args.es_patience, save_path=str(output_dir / "best_model.pt")
    )

    # Resume
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

    # 학습
    print(f"\n[3/4] Training (Tarabanis method: {args.epochs} epochs, Adam lr={args.lr}, "
          f"LR patience={args.lr_patience}/factor={args.lr_factor}, ES patience={args.es_patience})...")
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc, val_preds, val_probs, val_labels = evaluate(
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
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:2d} | "
              f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.3f} auroc={val_auroc:.4f} | "
              f"lr={current_lr:.1e} | {elapsed:.0f}s")

        # Checkpoint
        torch.save({
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "history": history, "best_val_loss": early_stopping.best_loss,
            "best_epoch": early_stopping.best_epoch, "es_counter": early_stopping.counter,
        }, str(ckpt_path))

        if early_stopping.step(val_loss, model, epoch):
            print(f"  Early stopping at epoch {epoch} (best: {early_stopping.best_epoch})")
            break

    # 평가
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
                                target_names=["Normal", "Hidden AF"]))
    print("Confusion Matrix:")
    print(confusion_matrix(test_labels, test_preds))

    # 결과 저장
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
