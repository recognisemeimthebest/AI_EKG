"""
부정맥 3-class 분류 모델 강화 — ResNet-34 + 6-limb lead

기존 CNN-TCN (3-lead, 126K params, Acc 90.6%) → ResNet-34 (6-limb, 21M params)
6-limb lead: Lead I, II → III, aVR, aVL, aVF 계산
"""
import argparse
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from resnet34_ecg import ResNet34ECGWithTabular
from dataset import ECGDataset, split_by_patient, compute_class_weights


class ECGDataset6Limb(ECGDataset):
    """ECGDataset + 6-limb lead 계산 (Lead I, II → III, aVR, aVL, aVF)"""

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        waveform = torch.tensor(self.h5["waveform"][real_idx], dtype=torch.float32)
        if self.lead_indices is not None:
            waveform = waveform[:, self.lead_indices]  # (5000, 2)

        lead_I = waveform[:, 0]
        lead_II = waveform[:, 1]
        lead_III = lead_II - lead_I
        aVR = -(lead_I + lead_II) / 2
        aVL = lead_I - lead_II / 2
        aVF = lead_II - lead_I / 2
        waveform = torch.stack([lead_I, lead_II, lead_III, aVR, aVL, aVF], dim=1)

        return waveform, self.numeric[idx], self.patient[idx], self.labels[idx]


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, save_path="best_model.pt"):
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
    all_preds, all_labels = [], []
    pbar = tqdm(loader, desc=f"         [{desc}]", leave=False,
                bar_format="{l_bar}{bar:30}{r_bar}")
    for wf, num, pat, lab in pbar:
        wf, num, pat, lab = wf.to(device), num.to(device), pat.to(device), lab.to(device)
        logits = model(wf, num, pat)
        loss = criterion(logits, lab)
        total_loss += loss.item() * lab.size(0)
        preds = logits.argmax(1)
        correct += (preds == lab).sum().item()
        total += lab.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(lab.cpu().numpy())
        pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.1%}")
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser(description="Arrhythmia 3-class (ResNet-34, 6-limb)")
    parser.add_argument("--data", default="g:/AIEKG/ml/data/ecg_preprocessed.h5")
    parser.add_argument("--output-dir", default="g:/AIEKG/ml/checkpoints/arrhythmia-resnet34-6limb")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--no-bp", action="store_true", default=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 데이터 (기존과 동일한 split)
    print("\n[1/4] 데이터 분할 (환자 단위)...")
    train_idx, val_idx, test_idx = split_by_patient(args.data)
    lead_indices = [0, 1]  # Lead I, II
    use_bp = not args.no_bp

    train_ds = ECGDataset6Limb(args.data, train_idx, lead_indices=lead_indices, use_bp=use_bp)
    val_ds = ECGDataset6Limb(args.data, val_idx, lead_indices=lead_indices, use_bp=use_bp)
    test_ds = ECGDataset6Limb(args.data, test_idx, lead_indices=lead_indices, use_bp=use_bp)

    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             pin_memory=True, num_workers=0)

    class_weights = compute_class_weights(args.data, train_idx)
    print(f"  Class weights: {class_weights.tolist()}")

    # 모델
    print("\n[2/4] ResNet-34 (6-limb lead, 3-class)...")
    n_numeric = train_ds.numeric.shape[1]
    n_patient = train_ds.patient.shape[1]
    model = ResNet34ECGWithTabular(
        n_leads=6, n_numeric=n_numeric, n_patient=n_patient,
        n_classes=3, dropout=args.dropout
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    early_stopping = EarlyStopping(
        patience=args.patience, save_path=str(output_dir / "best_model.pt")
    )

    # 학습
    print(f"\n[3/4] Training ({args.epochs} epochs, AdamW lr={args.lr})...")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr_now)

        print(f"  Epoch {epoch:2d} | train_loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.3f} | lr={lr_now:.1e} | {elapsed:.0f}s")

        scheduler.step(val_loss)

        torch.save({
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "history": history, "best_val_loss": early_stopping.best_loss,
            "best_epoch": early_stopping.best_epoch, "es_counter": early_stopping.counter,
            "args": vars(args),
        }, str(output_dir / "checkpoint.pt"))

        if early_stopping.step(val_loss, model, epoch):
            print(f"  Early stopping at epoch {epoch} (best: {early_stopping.best_epoch})")
            break

    # 테스트
    print(f"\n[4/4] Test evaluation (best epoch: {early_stopping.best_epoch})...")
    model.load_state_dict(torch.load(str(output_dir / "best_model.pt"), weights_only=True))
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device, desc="Test")

    class_names = ["Normal", "AFib", "Other"]
    report = classification_report(labels, preds, target_names=class_names)
    cm = confusion_matrix(labels, preds)

    print(f"\n  Test Accuracy: {test_acc:.1%}")
    print(report)
    print(f"Confusion Matrix:\n{cm}")

    results = {
        "test_loss": test_loss, "test_accuracy": test_acc,
        "best_epoch": early_stopping.best_epoch, "n_params": n_params,
        "args": vars(args), "history": history,
        "classification_report": report, "confusion_matrix": cm.tolist(),
    }
    with open(output_dir / "train_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {output_dir / 'train_results.json'}")


if __name__ == "__main__":
    main()
