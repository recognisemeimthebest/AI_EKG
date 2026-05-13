"""
SNN 임베딩 Linear Probe 평가 스크립트.

저장된 ECGEncoder 체크포인트를 로드하고 backbone을 freeze한 뒤,
LogisticRegression linear head로 AUROC/Accuracy/F1을 측정한다.

사용:
  python model/eval_linear_probe.py \
      --ckpt snn_tsne_10s.pt \
      --h5 E:/_archive/ecg_preprocessed.h5
"""
import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    classification_report, roc_curve
)
from sklearn.model_selection import train_test_split

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))

from model.snn_encoder import ECGEncoder
from model.train_snn_proto import (
    InMemoryECGDataset, SAMPLING_RATE, WINDOW_SAMPLES,
    _cv_worker, LEAD_II,
)


# ---------------------------------------------------------------------------
# Patient-level split 지원 버전의 데이터 로더
# ---------------------------------------------------------------------------
def sample_per_class_with_subjects(
    h5_path, per_class, seed=42, overdraw=1.5,
    n_workers=6, chunk_size=2000,
    normal_max=0.05, afib_min=0.15,
):
    """sample_per_class와 동일하지만 subject_id 배열도 함께 반환."""
    from concurrent.futures import ProcessPoolExecutor
    import time

    rng = np.random.default_rng(seed)
    draw = int(per_class * overdraw)

    with h5py.File(h5_path, "r") as f:
        all_labels  = f["label"][:]
        all_subjects = f["subject_id"][:]
        T = f["waveform"].shape[1]

    idx_normal = np.where(all_labels == 0)[0]
    idx_afib   = np.where(all_labels == 1)[0]
    sel_normal = rng.choice(idx_normal, size=min(draw, len(idx_normal)), replace=False)
    sel_afib   = rng.choice(idx_afib,   size=min(draw, len(idx_afib)),   replace=False)
    sel_idx    = np.concatenate([sel_normal, sel_afib])
    sel_labels = np.concatenate([
        np.zeros(len(sel_normal), dtype=np.int64),
        np.ones(len(sel_afib),   dtype=np.int64),
    ])
    order      = np.argsort(sel_idx)
    sel_idx    = sel_idx[order]
    sel_labels = sel_labels[order]
    sel_subj   = all_subjects[sel_idx]

    kept_wf, kept_lbl, kept_subj = [], [], []
    n_total = len(sel_idx)
    t0 = time.time()
    print(f"  filtering {n_total:,}  workers={n_workers}  chunk={chunk_size} ...")

    with h5py.File(h5_path, "r") as f:
        wf_dset = f["waveform"]
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for start in range(0, n_total, chunk_size):
                end       = min(start + chunk_size, n_total)
                chunk_idx = sel_idx[start:end]
                chunk_lbl = sel_labels[start:end]
                chunk_subj = sel_subj[start:end]
                raw = wf_dset[chunk_idx, :WINDOW_SAMPLES, :]
                lead2_list = [
                    (raw[i, :, LEAD_II].astype(np.float64), SAMPLING_RATE)
                    for i in range(len(chunk_idx))
                ]
                cvs = list(pool.map(_cv_worker, lead2_list, chunksize=32))
                for i, (cv, lbl, sid) in enumerate(zip(cvs, chunk_lbl, chunk_subj)):
                    if not np.isfinite(cv):
                        continue
                    if lbl == 0 and cv >= normal_max:
                        continue
                    if lbl == 1 and cv <= afib_min:
                        continue
                    wi  = raw[i, :, 0:1].astype(np.float32)
                    wii = raw[i, :, 1:2].astype(np.float32)
                    wf3 = np.concatenate([wi, wii, wii - wi], axis=1)
                    np.nan_to_num(wf3, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                    kept_wf.append(wf3)
                    kept_lbl.append(lbl)
                    kept_subj.append(sid)
                if (end % (chunk_size * 5) == 0 or end == n_total):
                    print(f"  {end:,}/{n_total:,}  ({time.time()-t0:.1f}s, kept={len(kept_wf)})")

    wf_out   = np.stack(kept_wf,  axis=0)
    lbl_out  = np.array(kept_lbl,  dtype=np.int64)
    subj_out = np.array(kept_subj, dtype=np.int64)
    print(f"  filtered: kept={len(wf_out)} "
          f"(Normal={int((lbl_out==0).sum())}, AFib={int((lbl_out==1).sum())}), "
          f"took {time.time()-t0:.1f}s")
    return wf_out, lbl_out, subj_out


def patient_level_split(h5_path, subject_ids, labels, seed=42,
                        train_ratio=0.6, val_ratio=0.2):
    """subject_id 단위로 train/val/test 분리. 동일 환자 레코딩은 같은 세트에."""
    rng = np.random.default_rng(seed)
    unique_subj = np.unique(subject_ids)
    rng.shuffle(unique_subj)

    n = len(unique_subj)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_subj = set(unique_subj[:n_train])
    val_subj   = set(unique_subj[n_train:n_train + n_val])
    test_subj  = set(unique_subj[n_train + n_val:])

    train_idx = np.where([s in train_subj for s in subject_ids])[0]
    val_idx   = np.where([s in val_subj   for s in subject_ids])[0]
    test_idx  = np.where([s in test_subj  for s in subject_ids])[0]

    print(f"  patients: train={len(train_subj):,}, val={len(val_subj):,}, test={len(test_subj):,}")
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# 임베딩 추출
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_embeddings(model: ECGEncoder, waveforms: np.ndarray,
                       labels: np.ndarray, device, batch_size: int = 256):
    model.eval()
    ds = InMemoryECGDataset(waveforms, labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    embs, labs = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        z = model(x)
        embs.append(z.cpu().numpy())
        labs.append(np.asarray(y))
    return np.concatenate(embs, axis=0), np.concatenate(labs, axis=0)


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # 체크포인트 로드
    print(f"[ckpt] {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    saved_args = ckpt.get("args", {})
    per_class = saved_args.get("per_class", args.per_class)
    overdraw  = saved_args.get("overdraw",  args.overdraw)
    seed      = saved_args.get("seed",      args.seed)
    dropout   = saved_args.get("dropout",   0.2)
    print(f"  per_class={per_class}, overdraw={overdraw}, seed={seed}")
    print(f"  best_val_loss={ckpt.get('best_val_loss', '?'):.4f}, "
          f"stopped={ckpt.get('stopped_epoch', '?')}")

    model = ECGEncoder(in_channels=3, embed_dim=128, dropout=dropout).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params={n_params:,}")

    # 데이터 로드 + RR-CV 필터
    workers = saved_args.get("workers", args.workers)
    print(f"[load+filter] {args.h5}  (workers={workers}, patient_split={args.patient_split})")
    wf, lbl, wf_subject_ids = sample_per_class_with_subjects(
        args.h5, per_class, seed=seed, overdraw=overdraw,
        n_workers=workers, chunk_size=args.chunk_size,
    )

    # 균형 맞춤
    keep_idx = []
    for cls in (0, 1):
        idx = np.where(lbl == cls)[0]
        keep_idx.append(idx[:per_class])
    keep_idx = np.concatenate(keep_idx)
    np.random.default_rng(seed).shuffle(keep_idx)
    wf, lbl, wf_subject_ids = wf[keep_idx], lbl[keep_idx], wf_subject_ids[keep_idx]
    print(f"  balanced: total={len(wf)}, Normal={int((lbl==0).sum())}, AFib={int((lbl==1).sum())}, "
          f"patients={len(np.unique(wf_subject_ids)):,}")

    # 분리 방식 선택
    if args.patient_split:
        train_idx, val_idx, test_idx = patient_level_split(
            args.h5, wf_subject_ids, lbl, seed=seed
        )
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(wf)), test_size=0.2, stratify=lbl, random_state=seed
        )
        train_idx, val_idx = train_test_split(
            train_idx, test_size=0.25, stratify=lbl[train_idx], random_state=seed
        )

    print(f"  split ({('patient' if args.patient_split else 'record')}): "
          f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # 임베딩 추출 (backbone freeze)
    print("[embed] ECGEncoder forward (frozen) ...")
    t0 = time.time()
    embs, labs = extract_embeddings(model, wf, lbl, device)
    print(f"  done in {time.time()-t0:.1f}s, shape={embs.shape}")

    X_train, y_train = embs[train_idx], labs[train_idx]
    X_val,   y_val   = embs[val_idx],   labs[val_idx]
    X_test,  y_test  = embs[test_idx],  labs[test_idx]

    # Linear Probe (Logistic Regression)
    print("[probe] LogisticRegression (max_iter=1000) ...")
    t0 = time.time()
    clf = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
    clf.fit(X_train, y_train)
    print(f"  trained in {time.time()-t0:.1f}s")

    def evaluate(X, y, split_name):
        prob = clf.predict_proba(X)[:, 1]
        pred = clf.predict(X)
        auroc = roc_auc_score(y, prob)
        acc   = accuracy_score(y, pred)
        f1    = f1_score(y, pred, average="binary")
        print(f"\n[{split_name}]")
        print(f"  AUROC={auroc:.4f}  Acc={acc:.4f}  F1={f1:.4f}")
        print(classification_report(y, pred, target_names=["Normal", "AFib"], digits=3))
        return auroc, acc, f1

    evaluate(X_train, y_train, "train")
    evaluate(X_val,   y_val,   "val")
    auroc, acc, f1 = evaluate(X_test,  y_test,  "test")

    print(f"\n[result] Test AUROC={auroc:.4f}  Acc={acc:.4f}  F1={f1:.4f}")
    print("[done]")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="snn_tsne_full.pt")
    p.add_argument("--h5", type=str,
                   default=os.environ.get("ECG_H5", r"E:\_archive\ecg_preprocessed.h5"))
    p.add_argument("--per-class", type=int, default=5000)
    p.add_argument("--overdraw", type=float, default=1.5)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--chunk-size", type=int, default=2000)
    p.add_argument("--patient-split", action="store_true",
                   help="환자 단위 분리 (기본값: 레코딩 단위)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
