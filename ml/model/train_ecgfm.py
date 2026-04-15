"""
ECG Foundation Model (ECG-FM) 기반 시퀀스 예측

백본 교체 전략:
  CNN-TCN 64dim → ECG-FM (MIMIC-IV 1.5M 사전학습) 768dim

입력 처리:
  5000 samples (10s) → seg1[:2500] + seg2[2500:] → 각 (2500,3)
  3-lead (I,II,III) → zero-pad → 12-lead (positions 0,1,2에 배치)
  ECG-FM → (T', 768) → mean pool → 768dim per segment → mean(seg1, seg2) → 768dim

WSL2 실행:
  source /mnt/g/AIEKG/venv_wsl/bin/activate
  cd /mnt/g/AIEKG/ml/model
  python train_ecgfm.py --ecg-fm-ckpt /mnt/g/AIEKG/ml/checkpoints/ecg-fm/mimic_iv_ecg_finetuned.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score
)
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from fairseq_signals.models import build_model_from_checkpoint
from sequence_dataset import (
    SequenceDataset, compute_class_weights_sequence, split_by_patient_sequence
)

# ─── 기본 경로 (WSL2 기준) ────────────────────────────────────────────────────
ECG_H5     = "/mnt/g/AIEKG/ml/data/ecg_preprocessed.h5"
SEQ_H5     = "/mnt/g/AIEKG/ml/data/ecg_sequence_15d_rhythm.h5"
CLINICAL_H5 = "/mnt/g/AIEKG/ml/data/clinical_features.h5"
ECG_FM_CKPT = "/mnt/g/AIEKG/ml/checkpoints/ecg-fm/mimic_iv_ecg_finetuned.pt"
OUTPUT_DIR  = "/mnt/g/AIEKG/ml/checkpoints/ecgfm-sequence"


# ─── 모델 정의 ────────────────────────────────────────────────────────────────

class TemporalBlock(nn.Module):
    """Causal Dilated Conv Block"""
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU()
        self.drop  = nn.Dropout(dropout)
        self.ds    = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.trim  = padding

    def forward(self, x):
        res = x
        o = self.conv1(x)
        if self.trim: o = o[:, :, :-self.trim]
        o = self.relu(self.bn1(o))
        o = self.drop(o)
        o = self.conv2(o)
        if self.trim: o = o[:, :, :-self.trim]
        o = self.relu(self.bn2(o))
        o = self.drop(o)
        if self.ds: res = self.ds(res)
        return self.relu(o + res)


class ECGFMSequenceModel(nn.Module):
    """
    ECG-FM 백본 + Temporal TCN 시퀀스 예측 모델

    각 ECG:
      (5000, 3) → 2 segments (2500, 3) → zero-pad to (12, 2500)
      → ECG-FM → (T', 768) → mean → 768dim
      → mean(seg1, seg2) → 768dim per ECG

    시퀀스:
      (B, S, 768+n_numeric+n_patient+1) → Temporal TCN → 2클래스
    """
    def __init__(self, ecgfm_model, ecgfm_dim=768,
                 n_numeric=4, n_patient=2, n_clinical=0,
                 tcn_channels=64, dropout=0.3, freeze_backbone=True):
        super().__init__()
        self.ecgfm = ecgfm_model
        self.freeze_backbone = freeze_backbone
        self.n_clinical = n_clinical
        self.ecgfm_dim = ecgfm_dim

        if freeze_backbone:
            for p in self.ecgfm.parameters():
                p.requires_grad = False

        # 입력 차원: ECG-FM 768 + numeric + patient + time_gap(1)
        feat_dim = ecgfm_dim + n_numeric + n_patient + 1

        # Temporal TCN
        self.temporal_tcn = nn.Sequential(
            TemporalBlock(feat_dim, tcn_channels, kernel_size=3, dilation=1, dropout=dropout),
            TemporalBlock(tcn_channels, tcn_channels, kernel_size=3, dilation=2, dropout=dropout),
            TemporalBlock(tcn_channels, tcn_channels, kernel_size=3, dilation=4, dropout=dropout),
        )

        # Late Fusion with clinical flags → classifier
        clf_in = tcn_channels + n_clinical
        self.classifier = nn.Sequential(
            nn.Linear(clf_in, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

    def _encode_ecg(self, waveform):
        """
        단일 ECG 인코딩
        waveform: (B, 5000, 3) — 3-lead, 10초
        returns: (B, 768)
        """
        B = waveform.shape[0]
        seg1 = waveform[:, :2500, :]     # (B, 2500, 3)
        seg2 = waveform[:, 2500:, :]     # (B, 2500, 3)

        def encode_seg(seg):
            # (B, 2500, 3) → zero-pad to (B, 12, 2500)
            padded = torch.zeros(B, 12, 2500, device=seg.device, dtype=seg.dtype)
            padded[:, :3, :] = seg.transpose(1, 2)  # I,II,III at 0,1,2

            # ECG-FM 추론
            # fairseq-signals wav2vec2 스타일: extract_features(source=...)
            result = self.ecgfm.extract_features(source=padded, padding_mask=None)
            x = result["encoder_out"]  # (B, T', 768)
            return x.mean(dim=1)      # (B, 768)

        if self.freeze_backbone:
            with torch.no_grad():
                f1 = encode_seg(seg1)
                f2 = encode_seg(seg2)
        else:
            f1 = encode_seg(seg1)
            f2 = encode_seg(seg2)

        return (f1 + f2) / 2.0   # (B, 768) — 두 세그먼트 평균

    def forward(self, waveforms, numerics, patients, mask, time_gaps,
                deltas=None, clinical=None):
        """
        waveforms:  (B, S, 5000, 3)
        numerics:   (B, S, n_numeric)
        patients:   (B, S, n_patient)
        mask:       (B, S)
        time_gaps:  (B, S)
        clinical:   (B, n_clinical) or None
        """
        B, S = waveforms.shape[:2]
        feats = []

        for t in range(S):
            ecg_feat = self._encode_ecg(waveforms[:, t])          # (B, 768)
            tg = time_gaps[:, t:t+1] / 30.0                       # (B, 1)
            f = torch.cat([ecg_feat, numerics[:, t],
                           patients[:, t], tg], dim=1)            # (B, feat_dim)
            f = f * mask[:, t:t+1]
            feats.append(f)

        # (B, feat_dim, S) → Temporal TCN
        feats = torch.stack(feats, dim=1).transpose(1, 2)         # (B, feat_dim, S)
        out = self.temporal_tcn(feats)                             # (B, tcn_ch, S)
        out = out[:, :, -1]                                        # (B, tcn_ch) 마지막 시점

        # Late Fusion
        if clinical is not None and clinical.shape[1] > 0:
            out = torch.cat([out, clinical], dim=1)

        return self.classifier(out)


# ─── 학습 유틸 ────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience=7, save_path="best_model.pt"):
        self.patience   = patience
        self.save_path  = save_path
        self.best_loss  = float("inf")
        self.best_epoch = 0
        self.counter    = 0

    def step(self, val_loss, model, epoch):
        if val_loss < self.best_loss:
            self.best_loss  = val_loss
            self.best_epoch = epoch
            self.counter    = 0
            torch.save(model.state_dict(), self.save_path)
            return False
        self.counter += 1
        return self.counter >= self.patience


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc=f"  Epoch {epoch} [Train]", leave=False)
    for waveforms, numerics, patients, mask, time_gaps, deltas, clinical, labels in pbar:
        waveforms  = waveforms.to(device)
        numerics   = numerics.to(device)
        patients   = patients.to(device)
        mask       = mask.to(device)
        time_gaps  = time_gaps.to(device)
        clinical   = clinical.to(device) if clinical.shape[1] > 0 else None
        labels     = labels.to(device)

        logits = model(waveforms, numerics, patients, mask, time_gaps,
                       clinical=clinical)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)
        pbar.set_postfix(acc=f"{correct/total:.1%}", loss=f"{total_loss/total:.4f}")

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="Val"):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_probs, all_labels = [], [], []
    pbar = tqdm(loader, desc=f"         [{desc}]", leave=False)
    for waveforms, numerics, patients, mask, time_gaps, deltas, clinical, labels in pbar:
        waveforms  = waveforms.to(device)
        numerics   = numerics.to(device)
        patients   = patients.to(device)
        mask       = mask.to(device)
        time_gaps  = time_gaps.to(device)
        clinical   = clinical.to(device) if clinical.shape[1] > 0 else None
        labels     = labels.to(device)

        logits = model(waveforms, numerics, patients, mask, time_gaps,
                       clinical=clinical)
        loss  = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()

        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)
        all_preds.extend(preds)
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix(acc=f"{correct/total:.1%}", loss=f"{total_loss/total:.4f}")

    return (total_loss / total, correct / total,
            np.array(all_preds), np.array(all_probs), np.array(all_labels))


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ECG-FM 시퀀스 예측")
    parser.add_argument("--ecg-data",    default=ECG_H5)
    parser.add_argument("--seq-data",    default=SEQ_H5)
    parser.add_argument("--clinical-data", default=CLINICAL_H5)
    parser.add_argument("--ecg-fm-ckpt", default=ECG_FM_CKPT)
    parser.add_argument("--output-dir",  default=OUTPUT_DIR)
    parser.add_argument("--batch-size",  type=int,   default=8)
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--dropout",     type=float, default=0.3)
    parser.add_argument("--patience",    type=int,   default=5)
    parser.add_argument("--tcn-channels",type=int,   default=64)
    parser.add_argument("--no-freeze",   action="store_true",
                        help="백본 파라미터도 학습 (기본: 고정)")
    parser.add_argument("--no-clinical", action="store_true",
                        help="clinical 피처 제외")
    parser.add_argument("--resume",      action="store_true")
    args = parser.parse_args()
    freeze_backbone = not args.no_freeze

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── 데이터 ────────────────────────────────────────────────────────────────
    print("\n[1/4] 데이터 로드...")
    lead_indices = [0, 1]   # I, II → add_lead3=True로 III 추가 → 3-lead

    train_idx, val_idx, test_idx = split_by_patient_sequence(args.seq_data)

    clinical_path = None if args.no_clinical else args.clinical_data

    train_ds = SequenceDataset(args.ecg_data, args.seq_data, train_idx,
                               lead_indices=lead_indices, clinical_h5_path=clinical_path)
    val_ds   = SequenceDataset(args.ecg_data, args.seq_data, val_idx,
                               lead_indices=lead_indices, clinical_h5_path=clinical_path)
    test_ds  = SequenceDataset(args.ecg_data, args.seq_data, test_idx,
                               lead_indices=lead_indices, clinical_h5_path=clinical_path)

    # add_lead3: I, II → III=II-I 계산 추가 → 3-lead
    train_ds.add_lead3 = True
    val_ds.add_lead3   = True
    test_ds.add_lead3  = True

    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              pin_memory=True, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              pin_memory=True, num_workers=0)

    class_weights = compute_class_weights_sequence(args.seq_data, train_idx)
    print(f"  Class weights: {class_weights.tolist()}")

    # ── 모델 ─────────────────────────────────────────────────────────────────
    print(f"\n[2/4] ECG-FM 로딩: {args.ecg_fm_ckpt}")
    result = build_model_from_checkpoint(args.ecg_fm_ckpt)
    ecgfm = result[0] if isinstance(result, tuple) else result
    ecgfm.eval()

    n_numeric  = train_ds.n_numeric
    n_patient  = train_ds.n_patient
    n_clinical = train_ds.n_clinical if not args.no_clinical else 0

    model = ECGFMSequenceModel(
        ecgfm_model    = ecgfm,
        ecgfm_dim      = 768,
        n_numeric      = n_numeric,
        n_patient      = n_patient,
        n_clinical     = n_clinical,
        tcn_channels   = args.tcn_channels,
        dropout        = args.dropout,
        freeze_backbone= freeze_backbone,
    ).to(device)

    n_params    = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  전체 파라미터: {n_params:,}")
    print(f"  학습 가능: {n_trainable:,} (backbone {'고정' if freeze_backbone else '학습'})")
    print(f"  n_numeric={n_numeric}, n_patient={n_patient}, n_clinical={n_clinical}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    stopper = EarlyStopping(
        patience=args.patience,
        save_path=str(output_dir / "best_model.pt")
    )

    start_epoch = 1
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_auroc": [], "lr": []
    }

    ckpt_path = output_dir / "checkpoint.pt"
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        history     = ckpt["history"]
        stopper.best_loss  = ckpt["best_val_loss"]
        stopper.best_epoch = ckpt["best_epoch"]
        stopper.counter    = ckpt["es_counter"]
        start_epoch = ckpt["epoch"] + 1
        print(f"\n  checkpoint 재개: epoch {ckpt['epoch']}")

    # ── 학습 ─────────────────────────────────────────────────────────────────
    print(f"\n[3/4] 학습 (epoch {start_epoch}~{args.epochs}, patience {args.patience})...")
    print(f"  Batch={args.batch_size}, LR={args.lr}, Backbone={'고정' if freeze_backbone else '학습'}")
    print(f"{'Epoch':>5} | {'Train Loss':>10} {'Acc':>9} | {'Val Loss':>10} {'Acc':>9} | "
          f"{'AUROC':>7} {'LR':>10} {'Time':>6}")
    print("-" * 80)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        vl_loss, vl_acc, _, vl_probs, vl_labels = evaluate(model, val_loader, criterion, device)
        vl_auroc = roc_auc_score(vl_labels, vl_probs)
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(f"{epoch:>5} | {tr_loss:>10.4f} {tr_acc:>9.1%} | "
              f"{vl_loss:>10.4f} {vl_acc:>9.1%} | "
              f"{vl_auroc:>7.4f} {lr:>10.6f} {elapsed:>5.0f}s")

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        history["val_auroc"].append(vl_auroc)
        history["lr"].append(lr)

        scheduler.step(vl_loss)
        torch.save({
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "history": history, "best_val_loss": stopper.best_loss,
            "best_epoch": stopper.best_epoch, "es_counter": stopper.counter,
        }, str(ckpt_path))

        if stopper.step(vl_loss, model, epoch):
            print(f"\nEarly stopping at epoch {epoch}. Best epoch: {stopper.best_epoch}")
            break

    # ── 테스트 ────────────────────────────────────────────────────────────────
    print(f"\n[4/4] 테스트 (best epoch: {stopper.best_epoch})...")
    model.load_state_dict(torch.load(str(output_dir / "best_model.pt"), weights_only=True))
    te_loss, te_acc, preds, probs, labels = evaluate(
        model, test_loader, criterion, device, desc="Test"
    )
    te_auroc = roc_auc_score(labels, probs)

    class_names = ["No Event", "Rhythm Abnormal"]
    report = classification_report(labels, preds, target_names=class_names)
    cm = confusion_matrix(labels, preds)

    print(f"\n  Test Loss:     {te_loss:.4f}")
    print(f"  Test Accuracy: {te_acc:.1%}")
    print(f"  Test AUROC:    {te_auroc:.4f}")
    print(f"\n{report}")
    print(f"  Confusion Matrix:\n{cm}")

    # 베이스라인 비교
    baseline_auroc = 0.7264
    delta = te_auroc - baseline_auroc
    sign = "+" if delta >= 0 else ""
    print(f"\n  베이스라인(CNN-TCN) AUROC: {baseline_auroc:.4f}")
    print(f"  ECG-FM AUROC:              {te_auroc:.4f} ({sign}{delta:.4f})")

    results = {
        "test_loss":               te_loss,
        "test_accuracy":           te_acc,
        "test_auroc":              te_auroc,
        "baseline_auroc":          baseline_auroc,
        "delta_vs_baseline":       delta,
        "best_epoch":              stopper.best_epoch,
        "n_params":                n_params,
        "n_trainable":             n_trainable,
        "args":                    vars(args),
        "history":                 history,
        "classification_report":   report,
        "confusion_matrix":        cm.tolist(),
    }
    with open(output_dir / "train_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  저장: {output_dir}")
    print("  완료!")


if __name__ == "__main__":
    main()
