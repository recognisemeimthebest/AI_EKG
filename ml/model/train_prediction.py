"""
30일 AFib 예측 모델 학습 스크립트

기존 CNN-TCN 부정맥 분류 모델의 피처 추출기를 재활용 (Transfer Learning)
분류 헤드만 2-class (정상 유지 / 30일 내 AFib)로 교체
"""
import argparse
import os
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tqdm import tqdm

from cnn_tcn import CNNTCN
from prediction_dataset import (
    PredictionDataset, split_by_patient_prediction, compute_class_weights_prediction
)


class AFibPredictor(nn.Module):
    """
    기존 CNN-TCN 피처 추출기 + 2-class 예측 헤드

    freeze_backbone=True면 CNN+TCN 가중치를 고정하고 헤드만 학습
    """

    def __init__(self, backbone: CNNTCN, n_numeric=4, n_patient=2, dropout=0.3, freeze_backbone=True):
        super().__init__()
        self.cnn = backbone.cnn
        self.tcn = backbone.tcn
        self.gap = backbone.gap

        if freeze_backbone:
            for param in self.cnn.parameters():
                param.requires_grad = False
            for param in self.tcn.parameters():
                param.requires_grad = False

        n_aux = n_numeric + n_patient
        self.aux_fc = nn.Sequential(
            nn.Linear(n_aux, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 2-class 예측 헤드
        self.classifier = nn.Sequential(
            nn.Linear(64 + 16, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

    def forward(self, waveform, numeric, patient):
        x = waveform.transpose(1, 2)
        x = self.cnn(x)
        x = self.tcn(x)
        x = self.gap(x)
        x = x.squeeze(-1)

        aux = torch.cat([numeric, patient], dim=1)
        aux = self.aux_fc(aux)

        combined = torch.cat([x, aux], dim=1)
        logits = self.classifier(combined)
        return logits


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
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"  Epoch {epoch} [Train]", leave=False,
                bar_format="{l_bar}{bar:30}{r_bar}")
    for waveform, numeric, patient, label in pbar:
        waveform = waveform.to(device)
        numeric = numeric.to(device)
        patient = patient.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        logits = model(waveform, numeric, patient)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * label.size(0)
        correct += (logits.argmax(1) == label).sum().item()
        total += label.size(0)
        pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.1%}")

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="Val"):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_probs = []
    all_labels = []

    pbar = tqdm(loader, desc=f"         [{desc}]", leave=False,
                bar_format="{l_bar}{bar:30}{r_bar}")
    for waveform, numeric, patient, label in pbar:
        waveform = waveform.to(device)
        numeric = numeric.to(device)
        patient = patient.to(device)
        label = label.to(device)

        logits = model(waveform, numeric, patient)
        loss = criterion(logits, label)

        total_loss += loss.item() * label.size(0)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(1)
        correct += (preds == label).sum().item()
        total += label.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # AFib 확률
        all_labels.extend(label.cpu().numpy())
        pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.1%}")

    return (total_loss / total, correct / total,
            np.array(all_preds), np.array(all_probs), np.array(all_labels))


def main():
    parser = argparse.ArgumentParser(description="30일 AFib 예측 모델 학습")
    parser.add_argument("--ecg-data", default="g:/AIEKG/ml/data/ecg_preprocessed.h5")
    parser.add_argument("--pred-data", default="g:/AIEKG/ml/data/ecg_prediction_30d.h5")
    parser.add_argument("--backbone", default="g:/AIEKG/ml/checkpoints/cnn-tcn-3lead/best_model.pt",
                        help="사전학습된 CNN-TCN 가중치")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--no-freeze", action="store_true",
                        help="백본도 함께 학습 (fine-tuning)")
    parser.add_argument("--resume", action="store_true",
                        help="checkpoint.pt에서 이어서 학습")
    parser.add_argument("--use-hrv", action="store_true",
                        help="HRV 피처 추가")
    parser.add_argument("--use-intervals", action="store_true",
                        help="ECG interval 피처 추가 (P-dur, PR, QTc, P-axis, QRS-T angle)")
    parser.add_argument("--shutdown", action="store_true")
    args = parser.parse_args()
    args.freeze_backbone = not args.no_freeze

    if args.output_dir is None:
        suffix = "prediction-15d" if args.freeze_backbone else "prediction-15d-finetune"
        if args.use_hrv:
            suffix += "-hrv"
        if args.use_intervals:
            suffix += "-intervals"
        args.output_dir = f"g:/AIEKG/ml/checkpoints/{suffix}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 데이터
    print("\n[1/4] 데이터 분할 중...")
    lead_indices = [0, 1]  # Lead I, II

    train_idx, val_idx, test_idx = split_by_patient_prediction(args.pred_data)
    train_ds = PredictionDataset(args.ecg_data, args.pred_data, train_idx, lead_indices=lead_indices, use_hrv=args.use_hrv, use_intervals=args.use_intervals)
    val_ds = PredictionDataset(args.ecg_data, args.pred_data, val_idx, lead_indices=lead_indices, use_hrv=args.use_hrv, use_intervals=args.use_intervals)
    test_ds = PredictionDataset(args.ecg_data, args.pred_data, test_idx, lead_indices=lead_indices, use_hrv=args.use_hrv, use_intervals=args.use_intervals)

    # Lead III 추가
    train_ds.add_lead3 = True
    val_ds.add_lead3 = True
    test_ds.add_lead3 = True

    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    class_weights = compute_class_weights_prediction(args.pred_data, train_idx)
    print(f"  Class weights: {class_weights.tolist()}")

    # 모델
    print("\n[2/4] 모델 생성 중...")
    n_numeric = train_ds.numeric.shape[1]
    backbone = CNNTCN(n_leads=3, n_numeric=n_numeric, n_classes=3, dropout=args.dropout)

    # 사전학습 가중치 로드
    print(f"  Backbone: {args.backbone}")
    state = torch.load(args.backbone, weights_only=True)
    # numeric 차원이 다르면 aux_fc 제외하고 로드
    mismatched = [k for k in state if k.startswith("aux_fc") and
                  state[k].shape != dict(backbone.named_parameters()).get(k, state[k]).shape]
    for k in mismatched:
        del state[k]
        print(f"  (skip: {k}, numeric dim mismatch)")
    backbone.load_state_dict(state, strict=False)

    model = AFibPredictor(
        backbone, n_numeric=n_numeric, dropout=args.dropout,
        freeze_backbone=args.freeze_backbone
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  전체 파라미터: {n_params:,}")
    print(f"  학습 가능: {n_trainable:,} (backbone {'고정' if args.freeze_backbone else '학습'})")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    if args.freeze_backbone:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=1e-4
        )
    else:
        # 차등 learning rate: 백본 lr/10, 헤드 lr
        backbone_params = list(model.cnn.parameters()) + list(model.tcn.parameters())
        head_params = list(model.aux_fc.parameters()) + list(model.classifier.parameters())
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": args.lr / 10},
            {"params": head_params, "lr": args.lr},
        ], weight_decay=1e-4)
        print(f"  차등 LR: 백본 {args.lr/10:.6f}, 헤드 {args.lr:.6f}")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    early_stopping = EarlyStopping(
        patience=args.patience, save_path=str(output_dir / "best_model.pt")
    )

    start_epoch = 1
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_auroc": [], "lr": []}

    # Resume
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
        print(f"\n  ✓ 체크포인트 복원: epoch {ckpt['epoch']}, "
              f"best_val_loss {early_stopping.best_loss:.4f} (epoch {early_stopping.best_epoch})")

    # 학습
    print(f"\n[3/4] 학습 시작 (에폭 {start_epoch}~{args.epochs}, patience {args.patience})...")
    print(f"  Batch: {args.batch_size}, LR: {args.lr}, Freeze: {args.freeze_backbone}")
    print("-" * 70)
    print(f"{'Epoch':>5} | {'Train Loss':>10} {'Train Acc':>10} | "
          f"{'Val Loss':>10} {'Val Acc':>10} | {'AUROC':>7} {'LR':>10} {'Time':>6}")
    print("-" * 70)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc, _, val_probs, val_labels = evaluate(model, val_loader, criterion, device)

        val_auroc = roc_auc_score(val_labels, val_probs)
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(f"{epoch:>5} | {train_loss:>10.4f} {train_acc:>9.1%} | "
              f"{val_loss:>10.4f} {val_acc:>9.1%} | {val_auroc:>7.4f} {lr:>10.6f} {elapsed:>5.0f}s")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auroc"].append(val_auroc)
        history["lr"].append(lr)

        scheduler.step(val_loss)

        torch.save({
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "history": history, "best_val_loss": early_stopping.best_loss,
            "best_epoch": early_stopping.best_epoch, "es_counter": early_stopping.counter,
        }, str(output_dir / "checkpoint.pt"))

        if early_stopping.step(val_loss, model, epoch):
            print(f"\nEarly stopping at epoch {epoch}. Best epoch: {early_stopping.best_epoch}")
            break

    # 테스트
    print(f"\n[4/4] 테스트 평가 중...")
    model.load_state_dict(torch.load(str(output_dir / "best_model.pt"), weights_only=True))
    test_loss, test_acc, preds, probs, labels = evaluate(model, test_loader, criterion, device, desc="Test")
    test_auroc = roc_auc_score(labels, probs)

    class_names = ["No AFib", "AFib <30d"]
    report = classification_report(labels, preds, target_names=class_names)
    cm = confusion_matrix(labels, preds)

    print(f"\n  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.1%}")
    print(f"  Test AUROC: {test_auroc:.4f}")
    print(f"\n  Classification Report:")
    print(report)
    print(f"  Confusion Matrix:")
    print(cm)

    # 저장
    results = {
        "test_loss": test_loss, "test_accuracy": test_acc, "test_auroc": test_auroc,
        "best_epoch": early_stopping.best_epoch, "n_params": n_params,
        "n_trainable": n_trainable, "args": vars(args), "history": history,
        "classification_report": report, "confusion_matrix": cm.tolist(),
    }
    with open(output_dir / "train_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  모델 저장: {output_dir / 'best_model.pt'}")
    print("  완료!")

    if args.shutdown:
        print("\n  60초 후 PC 종료... (취소: shutdown /a)")
        os.system("shutdown /s /t 60")


if __name__ == "__main__":
    main()
