"""
Morphology-focused 4-class ECG classifier (SupCon).

핵심 변경점 (기존 SNN과의 차이):
  1. RR-CV 필터 제거 → 의사 진단 라벨(report_0 기반) 직접 사용
     → 모델이 RR shortcut 못 쓰고 PQRST morphology를 강제로 학습
  2. 10초 ECG → 앞 5초 + 뒤 5초 split → 데이터 2배 (augmentation 아닌 단순 split)
  3. CNN-TCN backbone (cnn_tcn.py) — 부정맥 분류 93.6% 검증된 구조
  4. 4-class: Normal / AFib / OtherArr / Hyperkalemia

4 classes:
  0: Normal      — label==0 + K<5.0   (정상 진단 + 칼륨 정상)
  1: AFib        — label==1 + K<5.0   (AFib 진단 + 칼륨 정상)
  2: Other Arr   — label==2 + K<5.0   (기타 부정맥 + 칼륨 정상)
  3: Hyperkalemia — label==0 + K>=5.5  (정상 RR이지만 고칼륨)
"""
import argparse
import os
import sys
import time
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import label_binarize

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))

from model.cnn_tcn import CNNBlock, TCNBlock
from model.supcon_loss import SupConLoss


SAMPLING_RATE = 500
WINDOW_FULL   = 5000   # 10초
WINDOW_HALF   = 2500   # 5초
CLASS_NAMES = ["Normal", "AFib", "OtherArr", "HK"]


# ---------------------------------------------------------------------------
# 모델: CNN-TCN backbone + Projection head (L2 normalized)
# ---------------------------------------------------------------------------
class MorphologyEncoder(nn.Module):
    """
    CNN-TCN backbone (cnn_tcn.py) → 64-dim feature → projection → 128-dim L2 norm.

    입력: (B, T, 3) — 5초 = 2500 샘플
    출력: (B, 128) — L2 정규화된 임베딩 (SupCon용)
    """
    def __init__(self, in_channels: int = 3, embed_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.cnn = CNNBlock(in_channels=in_channels, dropout=dropout)
        self.tcn = TCNBlock(channels=64, n_layers=4, dropout=dropout)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Projection head (SupCon 관행)
        self.projection = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, embed_dim),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() != 3:
            raise ValueError(f"waveform must be 3D (B,T,C), got {waveform.shape}")
        x = waveform.transpose(1, 2)      # (B, C, T)
        x = self.cnn(x)                   # (B, 64, T')
        x = self.tcn(x)                   # (B, 64, T')
        x = self.gap(x).squeeze(-1)       # (B, 64)
        z = self.projection(x)            # (B, embed_dim)
        z = F.normalize(z, dim=1, p=2)    # 단위 구면 투영
        return z


# ---------------------------------------------------------------------------
# Dataset (in-memory)
# ---------------------------------------------------------------------------
class InMemoryECGDataset(Dataset):
    def __init__(self, wf: np.ndarray, lbl: np.ndarray):
        self.wf  = wf
        self.lbl = lbl

    def __len__(self): return len(self.wf)

    def __getitem__(self, i):
        return (torch.from_numpy(self.wf[i]).float(),
                int(self.lbl[i]))


# ---------------------------------------------------------------------------
# 4-class 인덱스 추출 (의사 진단 라벨 직접 사용)
# ---------------------------------------------------------------------------
def sample_4class_morphology(ecg_h5_path: str, hk_h5_path: str,
                              per_class: int = 8515,
                              k_normal_max: float = 5.0,
                              k_hk_min: float = 5.5,
                              time_diff_max_sec: int = 3600,
                              seed: int = 42):
    """
    의사 진단 label + K값으로 4-class 균형 샘플링.

    Returns:
        ecg_indices: (N,) ecg_preprocessed.h5 인덱스
        labels:      (N,) 0/1/2/3
        subject_ids: (N,) 환자 단위 분할용
    """
    rng = np.random.default_rng(seed)

    # 부정맥 라벨 (의사 진단)
    print("  로드: ecg_preprocessed.h5 label ...")
    with h5py.File(ecg_h5_path, "r") as f:
        arr_label = f["label"][:]
        all_subjects = f["subject_id"][:]

    # HK 정보 (K값 + 시간차)
    print("  로드: ecg_hyperkalemia_v2.h5 ...")
    with h5py.File(hk_h5_path, "r") as f:
        hk_indices = f["indices"][:]
        k_values   = f["k_value"][:]
        time_diffs = f["time_diff_sec"][:]

    # 시간차 1hr 이내 K 측정값만 신뢰
    tm = np.abs(time_diffs) < time_diff_max_sec
    hk_idx_filt = hk_indices[tm]
    k_filt      = k_values[tm]

    # ECG 인덱스 → K값 매핑 (없으면 NaN)
    k_lookup = np.full(len(arr_label), np.nan, dtype=np.float32)
    k_lookup[hk_idx_filt] = k_filt

    print(f"    K값 매핑된 ECG: {(~np.isnan(k_lookup)).sum():,}")

    # 4-class 마스크 (HK 정보 필요한 경우와 아닌 경우 분기)
    # Normal/AFib/Other: K<5.0 (확실히 HK 아님)
    # HK: K>=5.5 (확실한 HK)
    has_k = ~np.isnan(k_lookup)
    m0 = (arr_label == 0) & has_k & (k_lookup <  k_normal_max)
    m1 = (arr_label == 1) & has_k & (k_lookup <  k_normal_max)
    m2 = (arr_label == 2) & has_k & (k_lookup <  k_normal_max)
    m3 = (arr_label == 0) & has_k & (k_lookup >= k_hk_min)

    print("  원본 클래스별 가용 샘플:")
    for cls, m in enumerate([m0, m1, m2, m3]):
        print(f"    Class {cls} {CLASS_NAMES[cls]:10s}: {m.sum():,}")

    # 각 클래스 per_class만큼 추출
    sel_idx, sel_lbl, sel_sid = [], [], []
    for cls, m in enumerate([m0, m1, m2, m3]):
        avail = np.where(m)[0]
        n_pick = min(per_class, len(avail))
        chosen = rng.choice(avail, size=n_pick, replace=False)
        sel_idx.append(chosen)
        sel_lbl.append(np.full(n_pick, cls, dtype=np.int64))
        sel_sid.append(all_subjects[chosen])
        print(f"    pick Class {cls}: {n_pick:,}")

    sel_idx = np.concatenate(sel_idx)
    sel_lbl = np.concatenate(sel_lbl)
    sel_sid = np.concatenate(sel_sid)

    perm = rng.permutation(len(sel_idx))
    return sel_idx[perm], sel_lbl[perm], sel_sid[perm]


# ---------------------------------------------------------------------------
# 환자 단위 분할
# ---------------------------------------------------------------------------
def patient_level_split(subject_ids: np.ndarray, seed: int = 42,
                        train_frac: float = 0.7, val_frac: float = 0.15):
    rng = np.random.default_rng(seed)
    unique_subs = np.unique(subject_ids)
    rng.shuffle(unique_subs)
    n = len(unique_subs)
    n_tr = int(n * train_frac)
    n_va = int(n * val_frac)
    tr_subs = set(unique_subs[:n_tr].tolist())
    va_subs = set(unique_subs[n_tr:n_tr + n_va].tolist())

    tr_mask = np.array([s in tr_subs for s in subject_ids])
    va_mask = np.array([s in va_subs for s in subject_ids])
    te_mask = ~(tr_mask | va_mask)
    return tr_mask, va_mask, te_mask


# ---------------------------------------------------------------------------
# 청크 스트리밍 ECG 로드 + 5초 split (앞/뒤) → 데이터 2배
# ---------------------------------------------------------------------------
def load_ecg_split_chunked(ecg_h5_path: str, sel_idx: np.ndarray,
                            labels: np.ndarray, subject_ids: np.ndarray,
                            split_masks: tuple,
                            chunk_size: int = 2000):
    """
    각 ECG (10s, 5000 샘플) → 앞 5초 + 뒤 5초로 분리 → 데이터 2배.
    같은 ECG의 앞/뒤 5초는 같은 split (train/val/test)에 들어감.

    Returns:
        waveforms: (2N, 2500, 3) float32
        labels_2x: (2N,)
        masks_2x:  (tr_mask_2x, va_mask_2x, te_mask_2x)
    """
    n_total = len(sel_idx)
    t0 = time.time()

    # 정렬 인덱스 (h5py 순차 접근 최적화)
    sort_order = np.argsort(sel_idx)
    sorted_idx = sel_idx[sort_order]
    inv_order = np.argsort(sort_order)

    # 원본 N개 → 정렬 순서로 (N, 5000, 3)
    raw_sorted = np.empty((n_total, WINDOW_FULL, 3), dtype=np.float32)
    with h5py.File(ecg_h5_path, "r") as f:
        wf_dset = f["waveform"]
        for start in range(0, n_total, chunk_size):
            end = min(start + chunk_size, n_total)
            raw = wf_dset[sorted_idx[start:end], :WINDOW_FULL, :]
            wi  = raw[:, :, 0:1].astype(np.float32)
            wii = raw[:, :, 1:2].astype(np.float32)
            wf3 = np.concatenate([wi, wii, wii - wi], axis=2)
            np.nan_to_num(wf3, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            raw_sorted[start:end] = wf3
            if end % (chunk_size * 5) == 0 or end == n_total:
                print(f"    load {end:,}/{n_total:,}  ({time.time()-t0:.1f}s)")

    # 원래 순서 복원
    raw_all = raw_sorted[inv_order]  # (N, 5000, 3)
    del raw_sorted

    # 앞 5초 + 뒤 5초 → (2N, 2500, 3)
    front = raw_all[:, :WINDOW_HALF, :]            # 0~5s
    back  = raw_all[:, WINDOW_HALF:WINDOW_FULL, :]  # 5~10s
    wf_2x  = np.concatenate([front, back], axis=0)  # (2N, 2500, 3)
    lbl_2x = np.concatenate([labels, labels], axis=0)

    # split mask도 2x (앞/뒤 같은 split)
    tr_m, va_m, te_m = split_masks
    tr_2x = np.concatenate([tr_m, tr_m], axis=0)
    va_2x = np.concatenate([va_m, va_m], axis=0)
    te_2x = np.concatenate([te_m, te_m], axis=0)

    print(f"  split 완료: 원본 {n_total:,} → 5초 split 후 {len(wf_2x):,}")
    return wf_2x, lbl_2x, (tr_2x, va_2x, te_2x)


# ---------------------------------------------------------------------------
# 임베딩 추출
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_embeddings(model, wf, lbl, device, batch_size=256):
    model.eval()
    ds = InMemoryECGDataset(wf, lbl)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    embs = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        embs.append(model(x).cpu().numpy())
    return np.concatenate(embs, axis=0)


# ---------------------------------------------------------------------------
# 학습 루프
# ---------------------------------------------------------------------------
def train_supcon(model, train_loader, val_loader, device, args):
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.5, patience=args.patience // 2, min_lr=1e-6
    )
    criterion = SupConLoss(temperature=args.temperature)

    best_val = float("inf")
    best_state = None
    no_improve = 0
    stopped_epoch = args.epochs

    print(f"[train] epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}, T={args.temperature}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss, n_b = 0.0, 0
        t0 = time.time()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            z = model(x)
            loss = criterion(z, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()
            tr_loss += loss.item(); n_b += 1
        avg_tr = tr_loss / max(n_b, 1)

        model.eval()
        vl_loss, n_v = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                z = model(x)
                vl_loss += criterion(z, y).item(); n_v += 1
        avg_vl = vl_loss / max(n_v, 1)

        scheduler.step(avg_vl)
        cur_lr = optim.param_groups[0]["lr"]
        mark = "  *best*" if avg_vl < best_val - 1e-4 else f"  (no improve {no_improve+1}/{args.patience})"
        print(f"  ep {epoch:03d}/{args.epochs}  tr={avg_tr:.4f}  vl={avg_vl:.4f}  lr={cur_lr:.2e}  ({time.time()-t0:.1f}s){mark}")

        if avg_vl < best_val - 1e-4:
            best_val = avg_vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"[early stop] epoch {epoch}")
                stopped_epoch = epoch
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[restore] best val_loss={best_val:.4f}")
    return best_val, stopped_epoch


# ---------------------------------------------------------------------------
# Linear probe 평가 (multi-class)
# ---------------------------------------------------------------------------
def linear_probe_eval(model, wf_all, lbl_all, tr_mask, va_mask, te_mask, device, seed):
    print("\n[probe] 임베딩 추출 ...")
    emb_all = extract_embeddings(model, wf_all, lbl_all, device, batch_size=256)

    print("[probe] LogisticRegression (multinomial) 학습 ...")
    clf = LogisticRegression(max_iter=2000, random_state=seed, C=1.0,
                              class_weight="balanced", solver="lbfgs")
    clf.fit(emb_all[tr_mask], lbl_all[tr_mask])

    results = {}
    for name, mask in [("Train", tr_mask), ("Val", va_mask), ("Test", te_mask)]:
        X = emb_all[mask]; y = lbl_all[mask]
        pred = clf.predict(X)
        prob = clf.predict_proba(X)
        acc = accuracy_score(y, pred)
        f1_macro = f1_score(y, pred, average="macro", zero_division=0)

        y_bin = label_binarize(y, classes=[0, 1, 2, 3])
        auroc_macro = roc_auc_score(y_bin, prob, average="macro", multi_class="ovr")

        print(f"\n[{name}] n={len(y)}")
        print(f"  Acc={acc:.4f}  F1(macro)={f1_macro:.4f}  AUROC(macro)={auroc_macro:.4f}")
        print(classification_report(y, pred, target_names=CLASS_NAMES,
                                    digits=3, zero_division=0))
        print(f"  Confusion matrix:\n{confusion_matrix(y, pred)}")

        cls_auroc = {}
        for i, cname in enumerate(CLASS_NAMES):
            try:
                cls_auroc[cname] = float(roc_auc_score((y == i).astype(int), prob[:, i]))
            except Exception:
                cls_auroc[cname] = float("nan")
        print(f"  Per-class AUROC (one-vs-rest): {cls_auroc}")

        results[name] = dict(
            acc=float(acc), f1_macro=float(f1_macro),
            auroc_macro=float(auroc_macro),
            per_class_auroc=cls_auroc
        )

    return clf, results, emb_all


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    if device.type == "cuda":
        print(f"[gpu]    {torch.cuda.get_device_name(0)}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 4-class 인덱스 추출 (의사 진단 라벨 직접 사용)
    print("\n[1/5] 4-class 인덱스 추출 (RR-CV 필터 없음, 진단 라벨 사용) ...")
    sel_idx, lbl, sid = sample_4class_morphology(
        args.ecg_h5, args.hk_h5,
        per_class=args.per_class,
        k_normal_max=args.k_normal_max,
        k_hk_min=args.k_hk_min,
        time_diff_max_sec=args.time_diff_max,
        seed=args.seed,
    )
    print(f"  총 {len(sel_idx):,} ECG, 클래스 분포: {np.bincount(lbl)}")

    # 2. 환자 단위 분할 (ECG 단위, split 후 5초 split이 그대로 따라감)
    print("\n[2/5] 환자 단위 분할 (70/15/15) ...")
    tr_mask, va_mask, te_mask = patient_level_split(sid, seed=args.seed)
    print(f"  Train: {tr_mask.sum():,}, Val: {va_mask.sum():,}, Test: {te_mask.sum():,}")

    # 3. ECG 로드 + 5초 split (데이터 2배)
    print(f"\n[3/5] ECG 로드 + 앞/뒤 5초 split ...")
    wf_2x, lbl_2x, (tr_2x, va_2x, te_2x) = load_ecg_split_chunked(
        args.ecg_h5, sel_idx, lbl, sid,
        split_masks=(tr_mask, va_mask, te_mask),
        chunk_size=args.chunk_size,
    )
    print(f"  최종: shape={wf_2x.shape}, RAM={wf_2x.nbytes/1024**2:.0f} MB")
    print(f"  Train: {tr_2x.sum():,}, Val: {va_2x.sum():,}, Test: {te_2x.sum():,}")

    # 4. 학습
    print(f"\n[4/5] CNN-TCN + SupCon 학습 ...")
    ds_tr = InMemoryECGDataset(wf_2x[tr_2x], lbl_2x[tr_2x])
    ds_va = InMemoryECGDataset(wf_2x[va_2x], lbl_2x[va_2x])
    train_loader = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader   = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                              num_workers=0, drop_last=False)
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = MorphologyEncoder(in_channels=3, embed_dim=128, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  MorphologyEncoder params={n_params:,}")

    enc_path = out_dir / "encoder.pt"
    if args.skip_train and enc_path.exists():
        print(f"[skip-train] 기존 encoder 로드: {enc_path}")
        ckpt = torch.load(enc_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        best_val = ckpt.get("best_val_loss", 0.0)
        stopped_epoch = ckpt.get("stopped_epoch", 0)
    else:
        best_val, stopped_epoch = train_supcon(model, train_loader, val_loader, device, args)
        torch.save({
            "model": model.state_dict(),
            "args": vars(args),
            "best_val_loss": best_val,
            "stopped_epoch": stopped_epoch,
            "class_names": CLASS_NAMES,
        }, enc_path)
        print(f"[save] encoder → {enc_path}")

    # 5. Linear probe 평가
    print(f"\n[5/5] Linear probe (multinomial) 평가 ...")
    clf, results, emb_all = linear_probe_eval(
        model, wf_2x, lbl_2x, tr_2x, va_2x, te_2x, device, args.seed
    )

    import joblib
    clf_path = out_dir / "classifier.joblib"
    joblib.dump({
        "classifier": clf,
        "class_names": CLASS_NAMES,
        "k_normal_max": args.k_normal_max,
        "k_hk_min": args.k_hk_min,
        "encoder_ckpt": str(enc_path),
        "results": results,
    }, clf_path)
    print(f"[save] classifier → {clf_path}")

    json_path = out_dir / "train_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "best_val_loss": float(best_val),
            "stopped_epoch": int(stopped_epoch),
            "n_params": int(n_params),
            "class_names": CLASS_NAMES,
            "results": results,
            "args": vars(args),
        }, f, indent=2)
    print(f"[save] results → {json_path}")
    print("\n[done]")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ecg-h5", type=str,
                   default=os.environ.get("ECG_H5", r"E:\_archive\ecg_preprocessed.h5"))
    p.add_argument("--hk-h5", type=str,
                   default="G:/AIEKG/ml/data/ecg_hyperkalemia_v2.h5")
    p.add_argument("--out", type=str,
                   default="g:/AIEKG/ml/checkpoints/morphology-4cls")
    p.add_argument("--per-class",      type=int, default=8515)
    p.add_argument("--k-normal-max",   type=float, default=5.0)
    p.add_argument("--k-hk-min",       type=float, default=5.5)
    p.add_argument("--time-diff-max",  type=int, default=3600)
    p.add_argument("--epochs",         type=int, default=100)
    p.add_argument("--batch-size",     type=int, default=256)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--temperature",    type=float, default=0.1)
    p.add_argument("--dropout",        type=float, default=0.2)
    p.add_argument("--patience",       type=int, default=15)
    p.add_argument("--chunk-size",     type=int, default=2000)
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--skip-train",     action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
