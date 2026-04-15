"""
ECG-FM 헤드 학습 (사전 추출된 768-dim 피처 사용)

extract_ecgfm_features.py 실행 후 사용.
기존 train_sequence.py와 동일한 구조이나 백본이 없어 훨씬 빠름.

WSL2 또는 Windows에서 실행 가능:
  # Windows:
  g:/AIEKG/venv/Scripts/python.exe ml/model/train_ecgfm_head.py

  # WSL2:
  source /mnt/g/AIEKG/venv_wsl/bin/activate
  python /mnt/g/AIEKG/ml/model/train_ecgfm_head.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from sequence_dataset import split_by_patient_sequence, compute_class_weights_sequence

# ── 기본 경로 ─────────────────────────────────────────────────────────────────
_BASE      = "g:/AIEKG/ml"           # Windows / WSL2 모두 인식
SEQ_H5     = f"{_BASE}/data/ecg_sequence_15d_rhythm.h5"
FEAT_H5    = f"{_BASE}/data/ecgfm_features.h5"
CLINICAL_H5= f"{_BASE}/data/clinical_features.h5"
OUTPUT_DIR = f"{_BASE}/checkpoints/ecgfm-head"


# ─── Dataset ─────────────────────────────────────────────────────────────────

class ECGFMSequenceDataset(Dataset):
    """
    사전 추출된 ECG-FM 768-dim 피처 기반 시퀀스 Dataset

    각 샘플: 환자 ECG 시퀀스 → 리듬 이상 예측
    waveform 대신 pre-extracted 768-dim 벡터 사용
    """

    def __init__(self, seq_h5_path: str, feat_h5_path: str,
                 indices: np.ndarray = None, clinical_h5_path: str = None):

        # ── 시퀀스 메타 로드 ─────────────────────────────────────────────────
        with h5py.File(seq_h5_path, "r") as f:
            all_sequences  = f["sequences"][:]
            all_time_gaps  = (f["time_gaps"][:] if "time_gaps" in f
                              else np.zeros_like(all_sequences, dtype=np.float32))
            all_lengths    = f["seq_lengths"][:]
            all_labels     = f["pred_label"][:]
            all_sids       = f["subject_id"][:]
            self.max_seq_len = int(f.attrs["max_seq_len"])

        if indices is not None:
            self.sequences  = all_sequences[indices]
            self.time_gaps  = all_time_gaps[indices]
            self.lengths    = all_lengths[indices]
            self.labels     = torch.tensor(all_labels[indices], dtype=torch.long)
            self.subject_ids= all_sids[indices]
        else:
            self.sequences  = all_sequences
            self.time_gaps  = all_time_gaps
            self.lengths    = all_lengths
            self.labels     = torch.tensor(all_labels, dtype=torch.long)
            self.subject_ids= all_sids

        # ── ECG-FM 피처 캐시 ─────────────────────────────────────────────────
        all_ecg_indices = np.unique(self.sequences[self.sequences >= 0])
        print(f"    ECG-FM 피처 캐싱: {len(all_ecg_indices):,}개 로드 중...")

        with h5py.File(feat_h5_path, "r") as f:
            stored_indices = f["ecg_indices"][:]   # HDF5에 저장된 ECG 인덱스
            stored_features= f["features"][:]      # (N, 768) float16

        # ecg_idx → feature 벡터 매핑
        idx_to_pos = {int(idx): i for i, idx in enumerate(stored_indices)}
        self._feat_cache = {}
        missing = 0
        for ecg_idx in all_ecg_indices:
            pos = idx_to_pos.get(int(ecg_idx))
            if pos is not None:
                self._feat_cache[int(ecg_idx)] = (
                    stored_features[pos].astype(np.float32)
                )
            else:
                missing += 1
                self._feat_cache[int(ecg_idx)] = np.zeros(768, dtype=np.float32)

        if missing:
            print(f"    ⚠ 피처 없는 ECG: {missing}개 (zero 처리)")

        self.feat_dim = 768
        print(f"    캐싱 완료 ({len(self._feat_cache):,}개)")

        # ── numeric / patient 피처 캐시 (기존 HDF5에서) ─────────────────────
        ecg_h5_path = seq_h5_path.replace("ecg_sequence_15d_rhythm.h5",
                                           "ecg_preprocessed.h5")
        sorted_idx = np.sort(all_ecg_indices)
        with h5py.File(ecg_h5_path, "r") as f:
            raw_num = f["numeric_features"][sorted_idx]
            raw_pat = f["patient_features"][sorted_idx]

        self._numeric_cache = {}
        self._patient_cache = {}
        for i, idx in enumerate(sorted_idx):
            n = raw_num[i].astype(np.float32)
            p = raw_pat[i].astype(np.float32)
            np.nan_to_num(n, copy=False, nan=0.0)
            np.nan_to_num(p, copy=False, nan=0.0)
            self._numeric_cache[int(idx)] = n
            self._patient_cache[int(idx)] = p

        self.n_numeric = raw_num.shape[1]
        self.n_patient = raw_pat.shape[1]

        # ── clinical 피처 ────────────────────────────────────────────────────
        if clinical_h5_path is not None:
            with h5py.File(clinical_h5_path, "r") as f:
                clin_sids  = f["subject_id"][:]
                clin_flags = f["clinical_flags"][:]
            self._clinical_lookup = {
                int(sid): clin_flags[i].astype(np.float32)
                for i, sid in enumerate(clin_sids)
            }
            self.n_clinical = clin_flags.shape[1]
        else:
            self._clinical_lookup = None
            self.n_clinical = 0

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq      = self.sequences[idx]
        feats, numerics, patients, mask = [], [], [], []

        for i in range(self.max_seq_len):
            ecg_idx = int(seq[i])
            if ecg_idx >= 0:
                feats.append(torch.tensor(self._feat_cache[ecg_idx]))
                numerics.append(torch.tensor(self._numeric_cache[ecg_idx]))
                patients.append(torch.tensor(self._patient_cache[ecg_idx]))
                mask.append(1.0)
            else:
                feats.append(torch.zeros(self.feat_dim))
                numerics.append(torch.zeros(self.n_numeric))
                patients.append(torch.zeros(self.n_patient))
                mask.append(0.0)

        feats    = torch.stack(feats)     # (S, 768)
        numerics = torch.stack(numerics)  # (S, n_numeric)
        patients = torch.stack(patients)  # (S, n_patient)
        mask     = torch.tensor(mask, dtype=torch.float32)
        time_gaps= torch.tensor(self.time_gaps[idx], dtype=torch.float32)

        if self._clinical_lookup is not None:
            sid = int(self.subject_ids[idx])
            clinical = torch.tensor(
                self._clinical_lookup.get(sid, np.zeros(self.n_clinical, np.float32))
            )
        else:
            clinical = torch.zeros(0)

        return feats, numerics, patients, mask, time_gaps, clinical, self.labels[idx]


# ─── 모델 ─────────────────────────────────────────────────────────────────────

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU()
        self.drop  = nn.Dropout(dropout)
        self.ds    = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.trim  = pad

    def forward(self, x):
        res = x
        o = self.conv1(x)
        if self.trim: o = o[:, :, :-self.trim]
        o = self.relu(self.bn1(o)); o = self.drop(o)
        o = self.conv2(o)
        if self.trim: o = o[:, :, :-self.trim]
        o = self.relu(self.bn2(o)); o = self.drop(o)
        if self.ds: res = self.ds(res)
        return self.relu(o + res)


class ECGFMHead(nn.Module):
    """
    사전 추출된 768-dim ECG-FM 피처 + numeric/patient + time_gap
    → Temporal TCN → 2클래스 예측

    use_attention=True: 마지막 시점 대신 attention-weighted 집계
    """
    def __init__(self, feat_dim=768, n_numeric=4, n_patient=2, n_clinical=0,
                 tcn_channels=64, dropout=0.3, use_attention=False):
        super().__init__()
        self.n_clinical    = n_clinical
        self.use_attention = use_attention

        in_dim = feat_dim + n_numeric + n_patient + 1

        self.temporal_tcn = nn.Sequential(
            TemporalBlock(in_dim,       tcn_channels, 3, 1, dropout),
            TemporalBlock(tcn_channels, tcn_channels, 3, 2, dropout),
            TemporalBlock(tcn_channels, tcn_channels, 3, 4, dropout),
        )

        if use_attention:
            self.attn_fc = nn.Linear(tcn_channels, 1)

        clf_in = tcn_channels + n_clinical
        self.classifier = nn.Sequential(
            nn.Linear(clf_in, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

    def forward(self, feats, numerics, patients, mask, time_gaps, clinical=None):
        B, S = feats.shape[:2]

        tg  = (time_gaps / 30.0).unsqueeze(-1)
        x   = torch.cat([feats, numerics, patients, tg], dim=2)  # (B, S, in_dim)
        x   = x * mask.unsqueeze(-1)
        x   = x.transpose(1, 2)                                   # (B, in_dim, S)

        out = self.temporal_tcn(x)                                 # (B, tcn_ch, S)

        if self.use_attention:
            # attention score: 패딩 위치 -inf 마스킹 후 softmax
            scores  = self.attn_fc(out.transpose(1, 2))            # (B, S, 1)
            scores  = scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
            weights = torch.softmax(scores, dim=1)                 # (B, S, 1)
            out     = (out.transpose(1, 2) * weights).sum(dim=1)  # (B, tcn_ch)
        else:
            out = out[:, :, -1]                                    # 마지막 시점

        if clinical is not None and clinical.shape[1] > 0:
            out = torch.cat([out, clinical], dim=1)

        return self.classifier(out)


# ─── 학습 유틸 ────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience=7, save_path="best_model.pt"):
        self.patience = patience; self.save_path = save_path
        self.best_loss = float("inf"); self.best_epoch = 0; self.counter = 0

    def step(self, val_loss, model, epoch):
        if val_loss < self.best_loss:
            self.best_loss = val_loss; self.best_epoch = epoch
            self.counter = 0; torch.save(model.state_dict(), self.save_path)
            return False
        self.counter += 1
        return self.counter >= self.patience


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = correct = total = 0
    pbar = tqdm(loader, desc=f"  Epoch {epoch} [Train]", leave=False)
    for feats, numerics, patients, mask, time_gaps, clinical, labels in pbar:
        feats     = feats.to(device)
        numerics  = numerics.to(device)
        patients  = patients.to(device)
        mask      = mask.to(device)
        time_gaps = time_gaps.to(device)
        clinical  = clinical.to(device) if clinical.shape[1] > 0 else None
        labels    = labels.to(device)

        logits = model(feats, numerics, patients, mask, time_gaps, clinical)
        loss = criterion(logits, labels)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)
        pbar.set_postfix(acc=f"{correct/total:.1%}", loss=f"{total_loss/total:.4f}")
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="Val"):
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_probs, all_labels = [], [], []
    pbar = tqdm(loader, desc=f"         [{desc}]", leave=False)
    for feats, numerics, patients, mask, time_gaps, clinical, labels in pbar:
        feats     = feats.to(device)
        numerics  = numerics.to(device)
        patients  = patients.to(device)
        mask      = mask.to(device)
        time_gaps = time_gaps.to(device)
        clinical  = clinical.to(device) if clinical.shape[1] > 0 else None
        labels    = labels.to(device)

        logits = model(feats, numerics, patients, mask, time_gaps, clinical)
        loss   = criterion(logits, labels)
        probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds  = logits.argmax(1).cpu().numpy()

        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)
        all_preds.extend(preds); all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix(acc=f"{correct/total:.1%}", loss=f"{total_loss/total:.4f}")

    return (total_loss / total, correct / total,
            np.array(all_preds), np.array(all_probs), np.array(all_labels))


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ECG-FM Head 학습")
    parser.add_argument("--seq-data",      default=SEQ_H5)
    parser.add_argument("--feat-data",     default=FEAT_H5)
    parser.add_argument("--clinical-data", default=CLINICAL_H5)
    parser.add_argument("--output-dir",    default=OUTPUT_DIR)
    parser.add_argument("--batch-size",    type=int,   default=32)
    parser.add_argument("--epochs",        type=int,   default=30)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--dropout",       type=float, default=0.3)
    parser.add_argument("--patience",      type=int,   default=7)
    parser.add_argument("--tcn-channels",  type=int,   default=64)
    parser.add_argument("--no-clinical",   action="store_true")
    parser.add_argument("--use-attention", action="store_true",
                        help="마지막 시점 대신 attention-weighted 집계 사용")
    parser.add_argument("--resume",        action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── 데이터 ───────────────────────────────────────────────────────────────
    print("\n[1/4] 데이터 로드...")
    train_idx, val_idx, test_idx = split_by_patient_sequence(args.seq_data)
    clinical_path = None if args.no_clinical else args.clinical_data

    train_ds = ECGFMSequenceDataset(args.seq_data, args.feat_data, train_idx, clinical_path)
    val_ds   = ECGFMSequenceDataset(args.seq_data, args.feat_data, val_idx,   clinical_path)
    test_ds  = ECGFMSequenceDataset(args.seq_data, args.feat_data, test_idx,  clinical_path)
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
    print("\n[2/4] 모델 생성...")
    n_clinical = train_ds.n_clinical if not args.no_clinical else 0
    model = ECGFMHead(
        feat_dim      = 768,
        n_numeric     = train_ds.n_numeric,
        n_patient     = train_ds.n_patient,
        n_clinical    = n_clinical,
        tcn_channels  = args.tcn_channels,
        dropout       = args.dropout,
        use_attention = args.use_attention,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  파라미터: {n_params:,}")
    print(f"  n_numeric={train_ds.n_numeric}, n_patient={train_ds.n_patient}, n_clinical={n_clinical}")
    print(f"  use_attention={args.use_attention}, tcn_channels={args.tcn_channels}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    stopper = EarlyStopping(
        patience=args.patience, save_path=str(output_dir / "best_model.pt")
    )

    start_epoch = 1
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "val_auroc": [], "lr": []}

    ckpt_path = output_dir / "checkpoint.pt"
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        history = ckpt["history"]
        stopper.best_loss  = ckpt["best_val_loss"]
        stopper.best_epoch = ckpt["best_epoch"]
        stopper.counter    = ckpt["es_counter"]
        start_epoch = ckpt["epoch"] + 1
        print(f"  checkpoint 재개: epoch {ckpt['epoch']}")

    # ── 학습 ─────────────────────────────────────────────────────────────────
    print(f"\n[3/4] 학습 (epoch {start_epoch}~{args.epochs}, patience {args.patience})...")
    print(f"  Batch={args.batch_size}, LR={args.lr}, TCN channels={args.tcn_channels}")
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

        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss);   history["val_acc"].append(vl_acc)
        history["val_auroc"].append(vl_auroc); history["lr"].append(lr)

        scheduler.step(vl_loss)
        torch.save({
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "history": history, "best_val_loss": stopper.best_loss,
            "best_epoch": stopper.best_epoch, "es_counter": stopper.counter,
        }, str(ckpt_path))

        if stopper.step(vl_loss, model, epoch):
            print(f"\nEarly stopping at epoch {epoch}. Best: {stopper.best_epoch}")
            break

    # ── 테스트 ───────────────────────────────────────────────────────────────
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

    baseline = 0.7264
    delta = te_auroc - baseline
    sign = "+" if delta >= 0 else ""
    print(f"\n  베이스라인(CNN-TCN) AUROC: {baseline:.4f}")
    print(f"  ECG-FM Head AUROC:         {te_auroc:.4f} ({sign}{delta:.4f})")

    results = {
        "test_loss": te_loss, "test_accuracy": te_acc, "test_auroc": te_auroc,
        "baseline_auroc": baseline, "delta_vs_baseline": delta,
        "best_epoch": stopper.best_epoch, "n_params": n_params,
        "args": vars(args), "history": history,
        "classification_report": report, "confusion_matrix": cm.tolist(),
    }
    with open(output_dir / "train_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  저장: {output_dir}")
    print("  완료!")


if __name__ == "__main__":
    main()
