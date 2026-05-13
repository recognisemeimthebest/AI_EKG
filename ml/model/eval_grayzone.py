"""
회색지대(RR-CV 5~15%) 케이스에 대한 모델 성능 평가.

1. 클린 데이터(CV<5% or CV>15%)로 linear probe 학습
2. 회색지대(CV 5~15%) 케이스만 추출해서 평가
3. CV 5~10% / 10~15% 구간별 세부 분석

사용:
  python model/eval_grayzone.py --ckpt snn_tsne_full.pt --h5 E:/_archive/ecg_preprocessed.h5
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))

from model.snn_encoder import ECGEncoder
from model.train_snn_proto import InMemoryECGDataset, SAMPLING_RATE, WINDOW_SAMPLES, LEAD_II, _cv_worker


# ---------------------------------------------------------------------------
# 공통: 청크 스트리밍으로 RR-CV 계산 후 구간별 분류
# ---------------------------------------------------------------------------
def load_with_cv(h5_path, sel_idx, sel_labels, n_workers=6, chunk_size=2000):
    """선택된 인덱스의 waveform을 청크 단위로 읽고 RR-CV를 계산해서 반환.

    Returns:
        wf_out:   (N, WINDOW_SAMPLES, 3) float32
        lbl_out:  (N,) int64
        cv_out:   (N,) float64  — RR CV 값 (nan이면 계산 실패)
    """
    n_total = len(sel_idx)
    kept_wf, kept_lbl, kept_cv = [], [], []
    t0 = time.time()

    with h5py.File(h5_path, "r") as f:
        wf_dset = f["waveform"]
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for start in range(0, n_total, chunk_size):
                end       = min(start + chunk_size, n_total)
                chunk_idx = sel_idx[start:end]
                chunk_lbl = sel_labels[start:end]
                raw = wf_dset[chunk_idx, :WINDOW_SAMPLES, :]

                lead2_list = [
                    (raw[i, :, LEAD_II].astype(np.float64), SAMPLING_RATE)
                    for i in range(len(chunk_idx))
                ]
                cvs = list(pool.map(_cv_worker, lead2_list, chunksize=32))

                for i, (cv, lbl) in enumerate(zip(cvs, chunk_lbl)):
                    wi  = raw[i, :, 0:1].astype(np.float32)
                    wii = raw[i, :, 1:2].astype(np.float32)
                    wf3 = np.concatenate([wi, wii, wii - wi], axis=1)
                    np.nan_to_num(wf3, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                    kept_wf.append(wf3)
                    kept_lbl.append(lbl)
                    kept_cv.append(cv if np.isfinite(cv) else np.nan)

                if end % (chunk_size * 5) == 0 or end == n_total:
                    print(f"  {end:,}/{n_total:,}  ({time.time()-t0:.1f}s)")

    return (np.stack(kept_wf, axis=0),
            np.array(kept_lbl, dtype=np.int64),
            np.array(kept_cv,  dtype=np.float64))


# ---------------------------------------------------------------------------
# 임베딩 추출
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_embeddings(model, waveforms, labels, device, batch_size=256):
    model.eval()
    ds = InMemoryECGDataset(waveforms, labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    embs, labs = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        embs.append(model(x).cpu().numpy())
        labs.append(np.asarray(y))
    return np.concatenate(embs, axis=0), np.concatenate(labs, axis=0)


# ---------------------------------------------------------------------------
# 평가 출력
# ---------------------------------------------------------------------------
def evaluate(clf, X, y, label):
    prob = clf.predict_proba(X)[:, 1]
    pred = clf.predict(X)
    auroc = roc_auc_score(y, prob) if len(np.unique(y)) > 1 else float("nan")
    acc   = accuracy_score(y, pred)
    f1    = f1_score(y, pred, average="binary", zero_division=0)
    print(f"\n[{label}]  n={len(y)} (Normal={int((y==0).sum())}, AFib={int((y==1).sum())})")
    print(f"  AUROC={auroc:.4f}  Acc={acc:.4f}  F1={f1:.4f}")
    print(classification_report(y, pred, target_names=["Normal", "AFib"],
                                digits=3, zero_division=0))
    return auroc, acc, f1


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # 체크포인트 로드
    ckpt = torch.load(args.ckpt, map_location="cpu")
    saved = ckpt.get("args", {})
    per_class = saved.get("per_class", args.per_class)
    overdraw  = saved.get("overdraw",  1.0)
    seed      = saved.get("seed",      42)
    dropout   = saved.get("dropout",   0.2)
    print(f"[ckpt] {args.ckpt}  (per_class={per_class}, seed={seed})")

    model = ECGEncoder(in_channels=3, embed_dim=128, dropout=dropout).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 동일한 190K 풀에서 한 번만 로드 → CV로 clean/grayzone 분리
    rng = np.random.default_rng(seed)
    with h5py.File(args.h5, "r") as f:
        all_labels = f["label"][:]

    idx_normal = np.where(all_labels == 0)[0]
    idx_afib   = np.where(all_labels == 1)[0]
    draw = int(per_class * overdraw)
    sel_normal = rng.choice(idx_normal, size=min(draw, len(idx_normal)), replace=False)
    sel_afib   = rng.choice(idx_afib,   size=min(draw, len(idx_afib)),   replace=False)

    sel_all = np.concatenate([sel_normal, sel_afib])
    lbl_all = np.concatenate([np.zeros(len(sel_normal), dtype=np.int64),
                              np.ones(len(sel_afib),    dtype=np.int64)])
    order   = np.argsort(sel_all)
    sel_all = sel_all[order]
    lbl_all = lbl_all[order]

    # 한 번에 로드 + 전체 CV 계산
    print(f"\n[load] {len(sel_all):,} 샘플 로드 + RR-CV 계산 (clean + grayzone 통합) ...")
    wf_all, lbl_all, cv_all = load_with_cv(args.h5, sel_all, lbl_all,
                                            n_workers=args.workers)

    # CV 기준으로 분리
    clean_mask = (
        np.isfinite(cv_all) &
        (((lbl_all == 0) & (cv_all < 0.05)) |
         ((lbl_all == 1) & (cv_all > 0.15)))
    )
    gray_mask = (
        np.isfinite(cv_all) &
        (cv_all >= 0.05) & (cv_all <= 0.15)
    )

    wf_clean  = wf_all[clean_mask];  lbl_clean = lbl_all[clean_mask]
    wf_gray   = wf_all[gray_mask];   lbl_gray  = lbl_all[gray_mask]
    cv_gray   = cv_all[gray_mask]

    print(f"  clean:    {clean_mask.sum():,} (Normal={int((lbl_clean==0).sum())}, AFib={int((lbl_clean==1).sum())})")
    print(f"  grayzone: {gray_mask.sum():,} (Normal={int((lbl_gray==0).sum())}, AFib={int((lbl_gray==1).sum())})")

    # 클린 균형 맞춤
    keep = []
    for cls in (0, 1):
        idx = np.where(lbl_clean == cls)[0]
        keep.append(idx[:per_class])
    keep = np.concatenate(keep)
    np.random.default_rng(seed).shuffle(keep)
    wf_clean, lbl_clean = wf_clean[keep], lbl_clean[keep]

    # --- 임베딩 추출 ---
    print(f"\n[embed] 클린 {len(wf_clean):,}개 ...")
    emb_clean, lbl_c = extract_embeddings(model, wf_clean, lbl_clean, device)

    print(f"[embed] 회색지대 {len(wf_gray):,}개 ...")
    emb_gray, lbl_g = extract_embeddings(model, wf_gray, lbl_gray, device)

    # --- Linear Probe 학습 (클린 데이터만) ---
    print("\n[probe] LogisticRegression 학습 (클린 데이터) ...")
    clf = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
    clf.fit(emb_clean, lbl_c)

    # --- 평가 ---
    print("\n" + "="*60)
    evaluate(clf, emb_clean, lbl_c, "clean (학습 기준, 참고용)")

    evaluate(clf, emb_gray, lbl_g, "grayzone 전체 (CV 5~15%)")

    # 구간별 세부 분석
    for lo, hi, name in [(0.05, 0.10, "CV 5~10%"), (0.10, 0.15, "CV 10~15%")]:
        mask = (cv_gray >= lo) & (cv_gray < hi)
        if mask.sum() < 10:
            print(f"\n[{name}] 샘플 부족 ({mask.sum()}개), 생략")
            continue
        evaluate(clf, emb_gray[mask], lbl_g[mask], f"grayzone {name}")

    print("\n[done]")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="snn_tsne_full.pt")
    p.add_argument("--h5",   type=str,
                   default=os.environ.get("ECG_H5", r"E:\_archive\ecg_preprocessed.h5"))
    p.add_argument("--per-class",      type=int, default=100000)
    p.add_argument("--gray-per-class", type=int, default=50000,
                   help="회색지대 평가용 클래스당 샘플 수")
    p.add_argument("--workers",    type=int, default=6)
    p.add_argument("--chunk-size", type=int, default=2000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
