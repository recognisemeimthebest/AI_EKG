"""
시퀀스 TCN 예측 모델 학습

구조:
  각 ECG → 사전학습 CNN-TCN 백본 → 특징벡터 (64dim)
  시퀀스의 특징벡터들 → Temporal TCN → 2클래스 예측
"""
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from tqdm import tqdm
from pathlib import Path

from cnn_tcn import CNNTCN
from sequence_dataset import (
    SequenceDataset, split_by_patient_sequence, compute_class_weights_sequence
)


class TemporalBlock(nn.Module):
    """1D Causal Convolution Block for sequence TCN"""
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.trim = padding

    def forward(self, x):
        res = x
        out = self.conv1(x)
        if self.trim > 0:
            out = out[:, :, :-self.trim]
        out = self.relu(self.bn1(out))
        out = self.dropout(out)
        out = self.conv2(out)
        if self.trim > 0:
            out = out[:, :, :-self.trim]
        out = self.relu(self.bn2(out))
        out = self.dropout(out)
        if self.downsample is not None:
            res = self.downsample(res)
        return self.relu(out + res)


class SequenceTCN(nn.Module):
    """
    시퀀스 예측 모델:
    1. 사전학습 백본으로 각 ECG → 특징벡터 추출
    2. Temporal TCN으로 시퀀스 패턴 학습
    3. 2클래스 예측
    """
    def __init__(self, backbone, feature_dim=64, tcn_channels=32,
                 n_numeric=4, n_patient=2, n_clinical=0,
                 use_delta=False, dropout=0.3, freeze_backbone=True):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        self.n_clinical = n_clinical
        self.use_delta = use_delta

        if freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad = False

        # 백본의 GAP 출력 차원 = 64 (CNN-TCN의 tcn_channels)
        # + numeric + patient + time_gap(1) + delta(n_numeric) — clinical은 Late Fusion
        n_delta = n_numeric if use_delta else 0
        self.feature_dim = feature_dim + n_numeric + n_patient + 1 + n_delta

        # Temporal TCN: 시퀀스의 특징벡터 변화 학습
        self.temporal_tcn = nn.Sequential(
            TemporalBlock(self.feature_dim, tcn_channels, kernel_size=3, dilation=1, dropout=dropout),
            TemporalBlock(tcn_channels, tcn_channels, kernel_size=3, dilation=2, dropout=dropout),
        )

        # 분류 헤드 — Late Fusion: TCN 출력 + clinical → classifier
        classifier_input = tcn_channels + n_clinical
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 2),
        )

    def extract_features(self, waveform, numeric, patient):
        """단일 ECG에서 백본 특징 추출"""
        # waveform: (B, 5000, n_leads) → (B, n_leads, 5000)
        x = waveform.transpose(1, 2)
        x = self.backbone.cnn(x)
        x = self.backbone.tcn(x)
        x = self.backbone.gap(x).squeeze(-1)  # (B, 64)
        # numeric, patient 결합
        feat = torch.cat([x, numeric, patient], dim=1)  # (B, feature_dim)
        return feat

    def forward(self, waveforms, numerics, patients, mask, time_gaps,
                deltas=None, clinical=None):
        """
        waveforms: (B, seq_len, 5000, n_leads)
        numerics: (B, seq_len, n_numeric)
        patients: (B, seq_len, n_patient)
        mask: (B, seq_len)
        time_gaps: (B, seq_len) - 마지막 ECG 기준 일수 (30으로 나눠 정규화)
        deltas: (B, seq_len, n_numeric) - 연속 ECG 간 numeric 변화량
        clinical: (B, n_clinical) - 환자 단위 임상 플래그 (optional)
        """
        B, S = waveforms.shape[0], waveforms.shape[1]

        # 각 시점의 ECG에서 특징 추출
        features = []
        for t in range(S):
            w = waveforms[:, t]  # (B, 5000, n_leads)
            n = numerics[:, t]   # (B, n_numeric)
            p = patients[:, t]   # (B, n_patient)
            tg = (time_gaps[:, t:t+1] / 30.0)  # (B, 1) 정규화

            if self.freeze_backbone:
                with torch.no_grad():
                    feat = self.extract_features(w, n, p)
            else:
                feat = self.extract_features(w, n, p)

            # time_gap 결합
            feat = torch.cat([feat, tg], dim=1)

            # delta features 결합 (Suzuki 방식: 연속 ECG 간 변화량)
            if self.use_delta and deltas is not None:
                feat = torch.cat([feat, deltas[:, t]], dim=1)

            # 패딩 위치는 0으로
            feat = feat * mask[:, t:t+1]
            features.append(feat)

        # (B, seq_len, feature_dim) → (B, feature_dim, seq_len) for TCN
        features = torch.stack(features, dim=1)
        features = features.transpose(1, 2)

        # Temporal TCN
        out = self.temporal_tcn(features)  # (B, tcn_channels, seq_len)

        # 마지막 시점의 출력 사용 (가장 최근 ECG 기준 예측)
        out = out[:, :, -1]  # (B, tcn_channels)

        # Late Fusion: TCN 출력 뒤에 clinical 결합
        if clinical is not None and clinical.shape[1] > 0:
            out = torch.cat([out, clinical], dim=1)  # (B, tcn_channels + n_clinical)

        return self.classifier(out)


class EarlyStopping:
    def __init__(self, patience=7, save_path="best_model.pt"):
        self.patience = patience
        self.save_path = save_path
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.counter = 0

    def step(self, val_loss, model, epoch):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
            return False
        self.counter += 1
        return self.counter >= self.patience


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc=f"  Epoch {epoch} [Train]", leave=False)
    for waveforms, numerics, patients, mask, time_gaps, deltas, clinical, labels in pbar:
        waveforms = waveforms.to(device)
        numerics = numerics.to(device)
        patients = patients.to(device)
        mask = mask.to(device)
        time_gaps = time_gaps.to(device)
        deltas = deltas.to(device) if model.use_delta else None
        clinical = clinical.to(device) if clinical.shape[1] > 0 else None
        labels = labels.to(device)

        logits = model(waveforms, numerics, patients, mask, time_gaps, deltas, clinical)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(acc=f"{correct/total:.1%}", loss=f"{total_loss/total:.4f}")

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="Val"):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_probs, all_labels = [], [], []
    pbar = tqdm(loader, desc=f"         [{desc}]", leave=False)
    for waveforms, numerics, patients, mask, time_gaps, deltas, clinical, labels in pbar:
        waveforms = waveforms.to(device)
        numerics = numerics.to(device)
        patients = patients.to(device)
        mask = mask.to(device)
        time_gaps = time_gaps.to(device)
        deltas = deltas.to(device) if model.use_delta else None
        clinical = clinical.to(device) if clinical.shape[1] > 0 else None
        labels = labels.to(device)

        logits = model(waveforms, numerics, patients, mask, time_gaps, deltas, clinical)
        loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds)
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix(acc=f"{correct/total:.1%}", loss=f"{total_loss/total:.4f}")

    return (total_loss / total, correct / total,
            np.array(all_preds), np.array(all_probs), np.array(all_labels))


def main():
    parser = argparse.ArgumentParser(description="시퀀스 TCN 예측 모델")
    parser.add_argument("--ecg-data", default="g:/AIEKG/ml/data/ecg_preprocessed.h5")
    parser.add_argument("--seq-data", default="g:/AIEKG/ml/data/ecg_sequence_15d_rhythm.h5")
    parser.add_argument("--backbone", default="g:/AIEKG/ml/checkpoints/cnn-tcn-3lead/best_model.pt")
    parser.add_argument("--output-dir", default="g:/AIEKG/ml/checkpoints/sequence-15d-rhythm")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--no-freeze", action="store_true")
    parser.add_argument("--clinical-data", default=None,
                        help="Path to clinical_features.h5")
    parser.add_argument("--use-clinical", action="store_true",
                        help="Add clinical features (DM, HF, MI, AHT)")
    parser.add_argument("--use-delta", action="store_true",
                        help="Add delta features (Suzuki: numeric 변화량)")
    parser.add_argument("--use-interval", action="store_true",
                        help="Add ECG interval features (p_dur, PR, QTc, p_axis, QRS-T angle)")
    parser.add_argument("--use-hrv", action="store_true",
                        help="Add HRV features + delta (Gregoire 2025: RMSSD, SDNN, SD1, SD2 등 8개)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.freeze_backbone = not args.no_freeze

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 데이터
    print("\n[1/4] 데이터 로드...")
    lead_indices = [0, 1]
    train_idx, val_idx, test_idx = split_by_patient_sequence(args.seq_data)

    clinical_path = args.clinical_data if args.use_clinical else None
    if clinical_path:
        print(f"  Clinical features: {clinical_path}")

    train_ds = SequenceDataset(args.ecg_data, args.seq_data, train_idx,
                               lead_indices=lead_indices, clinical_h5_path=clinical_path,
                               use_interval=args.use_interval, use_hrv=args.use_hrv)
    val_ds = SequenceDataset(args.ecg_data, args.seq_data, val_idx,
                             lead_indices=lead_indices, clinical_h5_path=clinical_path,
                             use_interval=args.use_interval, use_hrv=args.use_hrv)
    test_ds = SequenceDataset(args.ecg_data, args.seq_data, test_idx,
                              lead_indices=lead_indices, clinical_h5_path=clinical_path,
                              use_interval=args.use_interval, use_hrv=args.use_hrv)
    train_ds.add_lead3 = True
    val_ds.add_lead3 = True
    test_ds.add_lead3 = True

    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             pin_memory=True, num_workers=0)

    class_weights = compute_class_weights_sequence(args.seq_data, train_idx)
    print(f"  Class weights: {class_weights.tolist()}")

    # 모델
    print("\n[2/4] 모델 생성...")
    n_numeric = train_ds.n_numeric
    n_patient = train_ds.n_patient
    # backbone은 원래 학습 시 n_numeric=4로 생성 (interval 미포함)
    backbone_n_numeric = 4
    backbone = CNNTCN(n_leads=3, n_numeric=backbone_n_numeric, n_classes=3, dropout=args.dropout)

    print(f"  Backbone: {args.backbone}")
    state = torch.load(args.backbone, weights_only=True)
    backbone.load_state_dict(state)

    n_clinical = train_ds.n_clinical
    model = SequenceTCN(
        backbone, feature_dim=64, tcn_channels=32,
        n_numeric=n_numeric, n_patient=n_patient, n_clinical=n_clinical,
        use_delta=args.use_delta,
        dropout=args.dropout, freeze_backbone=args.freeze_backbone
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  전체 파라미터: {n_params:,}")
    print(f"  학습 가능: {n_trainable:,} (backbone {'고정' if args.freeze_backbone else '학습'})")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    early_stopping = EarlyStopping(
        patience=args.patience, save_path=str(output_dir / "best_model.pt")
    )

    start_epoch = 1
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_auroc": [], "lr": []}

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
        print(f"\n  checkpoint: epoch {ckpt['epoch']}")

    # 학습
    print(f"\n[3/4] 학습 (epoch {start_epoch}~{args.epochs}, patience {args.patience})...")
    print(f"  Batch: {args.batch_size}, LR: {args.lr}, Freeze: {args.freeze_backbone}")
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
            print(f"\nEarly stopping at epoch {epoch}. Best: {early_stopping.best_epoch}")
            break

    # 테스트
    print(f"\n[4/4] 테스트...")
    model.load_state_dict(torch.load(str(output_dir / "best_model.pt"), weights_only=True))
    test_loss, test_acc, preds, probs, labels = evaluate(model, test_loader, criterion, device, desc="Test")
    test_auroc = roc_auc_score(labels, probs)

    class_names = ["No Event", "Rhythm Abnormal"]
    report = classification_report(labels, preds, target_names=class_names)
    cm = confusion_matrix(labels, preds)

    print(f"\n  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.1%}")
    print(f"  Test AUROC: {test_auroc:.4f}")
    print(f"\n{report}")
    print(f"  Confusion Matrix:\n{cm}")

    results = {
        "test_loss": test_loss, "test_accuracy": test_acc, "test_auroc": test_auroc,
        "best_epoch": early_stopping.best_epoch, "n_params": n_params,
        "n_trainable": n_trainable, "args": vars(args), "history": history,
        "classification_report": report, "confusion_matrix": cm.tolist(),
    }
    with open(output_dir / "train_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  저장: {output_dir}")
    print("  완료!")


if __name__ == "__main__":
    main()
