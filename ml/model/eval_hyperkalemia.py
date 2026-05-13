"""
SNN 임베딩 기반 고칼륨혈증(Hyperkalemia) 분류 평가.

전략:
  - AFib SNN encoder(frozen)에서 임베딩 추출
  - LogisticRegression으로 정상(K<5.0) vs 고칼륨(K>=6.0) 분류
  - 회색지대(K 5.0~6.0)는 학습 제외 (일시적 변동 가능성)
  - 환자 단위 분할 (data leakage 방지)
  - time_diff_sec < 3600 (1시간 이내) 만 사용 → 라벨 신뢰도 ↑

근거:
  - K>=5.5 mEq/L = 의학적 고칼륨혈증 기준
  - K>=6.0은 명확한 양성 (응급 처치 필요 임계)
  - 5.0~6.0은 경계 — 일시적 측정 오차/식이/약물 영향 가능

사용:
  python model/eval_hyperkalemia.py --ckpt snn_tsne_full.pt
"""
import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    classification_report, confusion_matrix, average_precision_score
)

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))

from model.snn_encoder import ECGEncoder
from model.train_snn_proto import InMemoryECGDataset, WINDOW_SAMPLES


# ---------------------------------------------------------------------------
# 라벨 정제
# ---------------------------------------------------------------------------
def select_clean_samples(hk_h5_path: str,
                         k_low: float = 5.0,
                         k_high: float = 6.0,
                         time_diff_max_sec: int = 3600):
    """
    K<k_low (clean normal) vs K>=k_high (clean hyperkalemia) 만 선택.
    time_diff_sec < time_diff_max_sec 필터 적용.

    Returns:
        ecg_indices: (N,) — ecg_preprocessed.h5에서의 인덱스
        hk_labels:   (N,) — 0=normal, 1=hyperkalemia
        subject_ids: (N,) — 환자 단위 분할용
        k_values:    (N,)
    """
    with h5py.File(hk_h5_path, "r") as f:
        indices  = f["indices"][:]
        k_value  = f["k_value"][:]
        sid      = f["subject_id"][:]
        time_d   = f["time_diff_sec"][:]

    # 시간차 필터
    time_mask = np.abs(time_d) < time_diff_max_sec

    # 라벨 정제: 명확한 양/음성만
    normal_mask = (k_value < k_low) & time_mask
    hk_mask     = (k_value >= k_high) & time_mask

    print(f"  필터링 결과:")
    print(f"    시간차 < {time_diff_max_sec}s : {time_mask.sum():,}")
    print(f"    Normal (K<{k_low}):              {normal_mask.sum():,}")
    print(f"    Hyperkalemia (K>={k_high}):      {hk_mask.sum():,}")
    print(f"    회색지대 ({k_low}<=K<{k_high}):     {((k_value >= k_low) & (k_value < k_high) & time_mask).sum():,} (제외)")

    sel_mask = normal_mask | hk_mask
    ecg_idx = indices[sel_mask]
    labels  = hk_mask[sel_mask].astype(np.int64)  # 1 = hyperkalemia
    subjs   = sid[sel_mask]
    kvals   = k_value[sel_mask]

    return ecg_idx, labels, subjs, kvals


# ---------------------------------------------------------------------------
# 환자 단위 분할
# ---------------------------------------------------------------------------
def patient_level_split(subject_ids: np.ndarray, seed: int = 42,
                        train_frac: float = 0.7, val_frac: float = 0.15):
    """환자 단위 60/20/20 분할 (data leakage 방지)."""
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
# 청크 스트리밍 ECG 로드 + 3-lead 변환
# ---------------------------------------------------------------------------
def load_ecg_chunked(ecg_h5_path: str, sel_idx: np.ndarray, chunk_size: int = 2000):
    """
    선택된 인덱스의 waveform을 청크 단위로 읽고 3-lead (I, II, II-I) 변환.

    Returns:
        wf: (N, WINDOW_SAMPLES, 3) float32
    """
    n_total = len(sel_idx)
    out = []
    t0 = time.time()

    # h5py는 인덱스 정렬 시 더 빠름 → 정렬 후 원위치 복원
    sort_order = np.argsort(sel_idx)
    sorted_idx = sel_idx[sort_order]
    inverse_order = np.argsort(sort_order)

    sorted_out = np.empty((n_total, WINDOW_SAMPLES, 3), dtype=np.float32)
    with h5py.File(ecg_h5_path, "r") as f:
        wf_dset = f["waveform"]
        for start in range(0, n_total, chunk_size):
            end = min(start + chunk_size, n_total)
            raw = wf_dset[sorted_idx[start:end], :WINDOW_SAMPLES, :]
            wi  = raw[:, :, 0:1].astype(np.float32)
            wii = raw[:, :, 1:2].astype(np.float32)
            wf3 = np.concatenate([wi, wii, wii - wi], axis=2)
            np.nan_to_num(wf3, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            sorted_out[start:end] = wf3
            if end % (chunk_size * 5) == 0 or end == n_total:
                print(f"    {end:,}/{n_total:,}  ({time.time()-t0:.1f}s)")

    return sorted_out[inverse_order]


# ---------------------------------------------------------------------------
# 임베딩 추출
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_embeddings(model, waveforms, labels, device, batch_size=256):
    model.eval()
    ds = InMemoryECGDataset(waveforms, labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    embs = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        embs.append(model(x).cpu().numpy())
    return np.concatenate(embs, axis=0)


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------
def evaluate(clf, X, y, label):
    prob = clf.predict_proba(X)[:, 1]
    pred = clf.predict(X)
    auroc = roc_auc_score(y, prob) if len(np.unique(y)) > 1 else float("nan")
    auprc = average_precision_score(y, prob) if len(np.unique(y)) > 1 else float("nan")
    acc   = accuracy_score(y, pred)
    f1    = f1_score(y, pred, average="binary", zero_division=0)
    print(f"\n[{label}]  n={len(y)} (Normal={int((y==0).sum())}, HK={int((y==1).sum())})")
    print(f"  AUROC={auroc:.4f}  AUPRC={auprc:.4f}  Acc={acc:.4f}  F1={f1:.4f}")
    print(classification_report(y, pred, target_names=["Normal", "Hyperkalemia"],
                                digits=3, zero_division=0))
    print(f"  Confusion matrix:\n{confusion_matrix(y, pred)}")
    return dict(auroc=auroc, auprc=auprc, acc=acc, f1=f1)


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # ----- 1. 체크포인트 로드 -----
    ckpt = torch.load(args.ckpt, map_location="cpu")
    saved = ckpt.get("args", {})
    dropout = saved.get("dropout", 0.2)
    seed    = saved.get("seed", 42)
    print(f"[ckpt] {args.ckpt}  (seed={seed}, dropout={dropout})")

    model = ECGEncoder(in_channels=3, embed_dim=128, dropout=dropout).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ----- 2. 라벨 정제 -----
    print(f"\n[1/5] Hyperkalemia 라벨 정제 ({args.hk_h5})...")
    ecg_idx, labels, subjs, kvals = select_clean_samples(
        args.hk_h5,
        k_low=args.k_low, k_high=args.k_high,
        time_diff_max_sec=args.time_diff_max
    )

    # 클래스 균형 맞춤 (Normal이 매우 많으니 hyperkalemia 수에 맞춤)
    n_hk = int((labels == 1).sum())
    n_normal_keep = min(int((labels == 0).sum()), n_hk * args.neg_ratio)
    rng = np.random.default_rng(seed)
    normal_indices = np.where(labels == 0)[0]
    hk_indices     = np.where(labels == 1)[0]
    keep_normal = rng.choice(normal_indices, size=n_normal_keep, replace=False)
    keep_all    = np.concatenate([keep_normal, hk_indices])
    rng.shuffle(keep_all)
    ecg_idx = ecg_idx[keep_all]
    labels  = labels[keep_all]
    subjs   = subjs[keep_all]
    kvals   = kvals[keep_all]

    print(f"  최종: 총 {len(labels):,} (Normal {int((labels==0).sum()):,}, HK {int((labels==1).sum()):,})")
    print(f"  K 분포: Normal min={kvals[labels==0].min():.2f}/max={kvals[labels==0].max():.2f}")
    print(f"           HK     min={kvals[labels==1].min():.2f}/max={kvals[labels==1].max():.2f}")

    # ----- 3. 환자 단위 분할 -----
    print(f"\n[2/5] 환자 단위 분할 (60/20/20)...")
    tr_mask, va_mask, te_mask = patient_level_split(subjs, seed=seed)
    print(f"  Train: {tr_mask.sum():,}  Val: {va_mask.sum():,}  Test: {te_mask.sum():,}")
    print(f"  unique patients: tr={len(set(subjs[tr_mask]))}, va={len(set(subjs[va_mask]))}, te={len(set(subjs[te_mask]))}")

    # ----- 4. ECG 로드 + 임베딩 추출 -----
    print(f"\n[3/5] ECG 로드 (chunk={args.chunk_size})...")
    waveforms = load_ecg_chunked(args.ecg_h5, ecg_idx, chunk_size=args.chunk_size)

    print(f"\n[4/5] SNN 임베딩 추출...")
    t0 = time.time()
    embeddings = extract_embeddings(model, waveforms, labels, device,
                                    batch_size=args.batch_size)
    print(f"  임베딩 shape: {embeddings.shape}  ({time.time()-t0:.1f}s)")
    del waveforms  # 메모리 절약

    # ----- 5. LogisticRegression 학습 + 평가 -----
    print(f"\n[5/5] LogisticRegression 학습 (클래스 가중 적용)...")
    clf = LogisticRegression(max_iter=2000, random_state=seed, C=args.C,
                              class_weight="balanced")
    clf.fit(embeddings[tr_mask], labels[tr_mask])

    print("\n" + "="*70)
    res_tr = evaluate(clf, embeddings[tr_mask], labels[tr_mask], "Train (참고용)")
    res_va = evaluate(clf, embeddings[va_mask], labels[va_mask], "Val")
    res_te = evaluate(clf, embeddings[te_mask], labels[te_mask], "Test (환자 분리)")

    # 분류기 저장
    if args.save_clf:
        import joblib
        save_path = Path(args.save_clf)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "classifier": clf,
            "k_low": args.k_low, "k_high": args.k_high,
            "time_diff_max": args.time_diff_max,
            "encoder_ckpt": args.ckpt,
            "results": {"train": res_tr, "val": res_va, "test": res_te},
        }, save_path)
        print(f"\n[save] 분류기 저장: {save_path}")

    # 베이스라인 비교 메시지
    print("\n" + "="*70)
    print(f"  Test AUROC: {res_te['auroc']:.4f}  AUPRC: {res_te['auprc']:.4f}")
    if res_te["auroc"] >= 0.80:
        print(f"  [GREAT] 임베딩이 hyperkalemia 패턴도 잘 포착함! Frozen encoder 충분")
    elif res_te["auroc"] >= 0.70:
        print(f"  [OK] 어느 정도 작동. encoder fine-tune으로 더 올릴 수 있음")
    elif res_te["auroc"] >= 0.60:
        print(f"  [WEAK] 임베딩에 hyperkalemia 시그널 부족. fine-tune 권장")
    else:
        print(f"  [FAIL] 별도 SNN 학습 필요")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",   type=str, default="snn_tsne_full.pt",
                   help="AFib SNN encoder 체크포인트")
    p.add_argument("--ecg-h5", type=str,
                   default=os.environ.get("ECG_H5", r"E:\_archive\ecg_preprocessed.h5"))
    p.add_argument("--hk-h5",  type=str,
                   default="G:/AIEKG/ml/data/ecg_hyperkalemia_v2.h5")
    p.add_argument("--k-low",  type=float, default=5.0, help="Normal 상한 (K<k_low)")
    p.add_argument("--k-high", type=float, default=6.0, help="HK 하한 (K>=k_high)")
    p.add_argument("--time-diff-max", type=int, default=3600,
                   help="ECG-혈액검사 시간차 최대 (초)")
    p.add_argument("--neg-ratio",  type=int, default=3,
                   help="Normal:HK 비율 (기본 3:1, balanced 학습용)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--chunk-size", type=int, default=2000)
    p.add_argument("--C",          type=float, default=1.0,
                   help="LogisticRegression 정규화 강도 (작을수록 강함)")
    p.add_argument("--save-clf",   type=str, default="g:/AIEKG/ml/checkpoints/snn-hyperkalemia/classifier.joblib",
                   help="분류기 저장 경로 (.joblib)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
