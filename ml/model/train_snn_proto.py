"""
SNN/SupCon 프로토타입 학습 스크립트.

목적:
  - HDF5에서 AFib/Normal 각 N건 샘플링.
  - neurokit2로 R-peak를 검출, RR-interval CV 기반 라벨 노이즈 필터링:
      * Normal(0): RR CV < 5%   → 사용
      * AFib(1):   RR CV > 15%  → 사용
      * 그 외(5~15%): 학습에서 제외
  - 5000샘플(10초 전체) × 3-lead(I, II, III=II-I) 사용.
  - SupCon Loss로 임베딩 학습 후 t-SNE 시각화 PNG 저장.

사용:
  python -m model.train_snn_proto \
      --h5 E:/_archive/ecg_preprocessed.h5 \
      --out ./snn_tsne_proto.png \
      --per-class 5000 --epochs 100 --patience 10

주의: 프로토타입이므로 환자 단위 분할 등 실험 위생은 생략. 첫 검증이 목적.
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
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# 모델 모듈 import (스크립트로 실행 시 경로 보강)
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))

from model.snn_encoder import ECGEncoder  # noqa: E402
from model.supcon_loss import SupConLoss  # noqa: E402

SAMPLING_RATE = 500   # ECG 전처리 스펙: 500Hz
WINDOW_SAMPLES = 5000  # 10초 전체
LEAD_I, LEAD_II = 0, 1


# ---------------------------------------------------------------------------
# RR CV 계산 및 라벨 필터링
# ---------------------------------------------------------------------------
def _cv_worker(args):
    """ProcessPoolExecutor용 최상위 함수 (pickle 필요)."""
    sig, fs = args
    return rr_cv_from_lead2(sig, fs)


def rr_cv_from_lead2(signal_lead2: np.ndarray, fs: int = SAMPLING_RATE) -> float:
    """Lead II 1D 신호에서 R-peak 검출 → RR 간격의 변동계수(CV) 반환.

    Returns:
        RR CV (std/mean). 검출 실패 또는 R-peak<3개면 np.nan.
    """
    try:
        import neurokit2 as nk
    except ImportError as e:
        raise RuntimeError("neurokit2가 필요합니다. venv에 설치되어야 합니다.") from e

    try:
        # clean → 다양한 필터링/정규화 적용. 짧은 5초 윈도우에도 동작.
        cleaned = nk.ecg_clean(signal_lead2, sampling_rate=fs, method="neurokit")
        _, info = nk.ecg_peaks(cleaned, sampling_rate=fs, method="neurokit", correct_artifacts=True)
        rpeaks = info.get("ECG_R_Peaks", np.array([]))
        if rpeaks is None or len(rpeaks) < 3:
            return float("nan")
        rr = np.diff(rpeaks).astype(np.float64) / fs  # 초 단위
        if rr.mean() <= 0:
            return float("nan")
        return float(rr.std() / rr.mean())
    except Exception:
        return float("nan")


def filter_by_rr_cv(
    waveforms: np.ndarray,
    labels: np.ndarray,
    fs: int = SAMPLING_RATE,
    normal_max: float = 0.05,
    afib_min: float = 0.15,
    verbose: bool = True,
):
    """RR CV 기반 라벨 노이즈 필터.

    Args:
        waveforms: (N, T, C) — Lead II는 index 1.
        labels:    (N,)      — 0=Normal, 1=AFib (그 외 라벨은 사전 제외 가정).
    Returns:
        (kept_waveforms, kept_labels, stats_dict)
    """
    n = len(waveforms)
    keep = np.zeros(n, dtype=bool)
    cvs = np.full(n, np.nan, dtype=np.float64)

    t0 = time.time()
    for i in range(n):
        sig = waveforms[i, :, LEAD_II].astype(np.float64)
        cv = rr_cv_from_lead2(sig, fs=fs)
        cvs[i] = cv
        if not np.isfinite(cv):
            continue
        if labels[i] == 0 and cv < normal_max:
            keep[i] = True
        elif labels[i] == 1 and cv > afib_min:
            keep[i] = True
        if verbose and (i + 1) % 500 == 0:
            print(f"  RR-CV {i+1}/{n}  ({time.time()-t0:.1f}s, kept={keep.sum()})")

    stats = {
        "n_total": n,
        "n_kept": int(keep.sum()),
        "n_normal_kept": int(((labels == 0) & keep).sum()),
        "n_afib_kept": int(((labels == 1) & keep).sum()),
        "n_invalid_rr": int(np.sum(~np.isfinite(cvs))),
        "elapsed_sec": time.time() - t0,
    }
    return waveforms[keep], labels[keep], stats


# ---------------------------------------------------------------------------
# 샘플링 + RR-CV 필터링: 청크 스트리밍 (OOM 방지)
# ---------------------------------------------------------------------------
def sample_per_class(
    h5_path: str,
    per_class: int,
    seed: int = 42,
    overdraw: float = 1.5,
    n_workers: int = 6,
    chunk_size: int = 2000,
    normal_max: float = 0.05,
    afib_min: float = 0.15,
    verbose: bool = True,
):
    """HDF5에서 청크 단위로 읽으며 RR-CV 필터링까지 수행.

    전체를 RAM에 올리지 않고 chunk_size씩 읽어 처리 -> OOM 방지.
    chunk_size=2000 기준 피크 메모리: ~480MB.
    n_workers=6: Ryzen 9700X (8코어) 기준 권장값.
    """
    rng = np.random.default_rng(seed)
    draw = int(per_class * overdraw)

    with h5py.File(h5_path, "r") as f:
        all_labels = f["label"][:]
        T = f["waveform"].shape[1]
        if T < WINDOW_SAMPLES:
            raise ValueError(f"waveform length {T} < required {WINDOW_SAMPLES}")

    idx_normal = np.where(all_labels == 0)[0]
    idx_afib   = np.where(all_labels == 1)[0]
    if verbose:
        print(f"  available: Normal={len(idx_normal):,}, AFib={len(idx_afib):,}")

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

    kept_wf, kept_lbl = [], []
    n_total   = len(sel_idx)
    n_invalid = 0
    t0 = time.time()

    if verbose:
        print(f"  filtering {n_total:,} samples  workers={n_workers}  chunk={chunk_size} ...")

    with h5py.File(h5_path, "r") as f:
        wf_dset = f["waveform"]
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for start in range(0, n_total, chunk_size):
                end       = min(start + chunk_size, n_total)
                chunk_idx = sel_idx[start:end]
                chunk_lbl = sel_labels[start:end]

                # 청크 로드: (chunk, WINDOW_SAMPLES, 12)
                raw = wf_dset[chunk_idx, :WINDOW_SAMPLES, :]

                lead2_list = [
                    (raw[i, :, LEAD_II].astype(np.float64), SAMPLING_RATE)
                    for i in range(len(chunk_idx))
                ]
                cvs = list(pool.map(_cv_worker, lead2_list, chunksize=32))

                for i, (cv, lbl) in enumerate(zip(cvs, chunk_lbl)):
                    if not np.isfinite(cv):
                        n_invalid += 1
                        continue
                    if lbl == 0 and cv >= normal_max:
                        continue
                    if lbl == 1 and cv <= afib_min:
                        continue
                    wi   = raw[i, :, 0:1].astype(np.float32)
                    wii  = raw[i, :, 1:2].astype(np.float32)
                    wiii = wii - wi
                    wf3  = np.concatenate([wi, wii, wiii], axis=1)  # (WINDOW_SAMPLES, 3)
                    np.nan_to_num(wf3, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                    kept_wf.append(wf3)
                    kept_lbl.append(lbl)

                if verbose and (end % (chunk_size * 5) == 0 or end == n_total):
                    elapsed = time.time() - t0
                    print(f"  {end:,}/{n_total:,}  ({elapsed:.1f}s, kept={len(kept_wf)})")

    elapsed = time.time() - t0
    wf_out  = np.stack(kept_wf, axis=0)
    lbl_out = np.array(kept_lbl, dtype=np.int64)
    n_normal = int((lbl_out == 0).sum())
    n_afib   = int((lbl_out == 1).sum())
    if verbose:
        print(f"  filtered: kept={len(wf_out)}  "
              f"(Normal={n_normal}, AFib={n_afib}), "
              f"invalid_rr={n_invalid}, took {elapsed:.1f}s")
    return wf_out, lbl_out


# ---------------------------------------------------------------------------
# Dataset (in-memory; 프로토타입 규모이므로 HDF5 재오픈 불필요)
# ---------------------------------------------------------------------------
class InMemoryECGDataset(Dataset):
    def __init__(self, waveforms: np.ndarray, labels: np.ndarray):
        # 채널별 z-score 정규화 (per-sample, per-lead). 단순하지만 강건.
        mean = waveforms.mean(axis=1, keepdims=True)
        std = waveforms.std(axis=1, keepdims=True) + 1e-6
        self.x = ((waveforms - mean) / std).astype(np.float32)
        self.y = labels.astype(np.int64)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), int(self.y[i])


# ---------------------------------------------------------------------------
# t-SNE 시각화
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_embeddings(model: ECGEncoder, ds: Dataset, device, batch_size: int = 256):
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    embs, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        z = model(x)
        embs.append(z.cpu().numpy())
        labels.append(np.asarray(y))
    return np.concatenate(embs, axis=0), np.concatenate(labels, axis=0)


def save_tsne(embs: np.ndarray, labels: np.ndarray, out_path: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    # 너무 많으면 t-SNE가 느리므로 최대 4000으로 균형 샘플링
    cap = 4000
    if len(embs) > cap:
        rng = np.random.default_rng(0)
        idx0 = np.where(labels == 0)[0]
        idx1 = np.where(labels == 1)[0]
        k = cap // 2
        idx0 = rng.choice(idx0, size=min(k, len(idx0)), replace=False)
        idx1 = rng.choice(idx1, size=min(k, len(idx1)), replace=False)
        sel = np.concatenate([idx0, idx1])
        embs = embs[sel]
        labels = labels[sel]

    tsne = TSNE(n_components=2, perplexity=30, init="pca",
                learning_rate="auto", random_state=42)
    proj = tsne.fit_transform(embs)

    plt.figure(figsize=(8, 7))
    for cls, name, color in [(0, "Normal", "#1f77b4"), (1, "AFib", "#d62728")]:
        m = labels == cls
        plt.scatter(proj[m, 0], proj[m, 1], s=6, alpha=0.55,
                    label=f"{name} (n={int(m.sum())})", c=color)
    plt.legend(loc="best")
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[save] t-SNE → {out_path}")


# ---------------------------------------------------------------------------
# 학습
# ---------------------------------------------------------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    print(f"[load] HDF5: {args.h5}  "
          f"(per-class={args.per_class}, overdraw={args.overdraw}, workers={args.workers})")
    wf, lbl = sample_per_class(
        args.h5, args.per_class, seed=args.seed, overdraw=args.overdraw,
        n_workers=args.workers, chunk_size=args.chunk_size,
    )

    # 클래스별로 per_class까지만 잘라 균형 맞춤
    keep_idx = []
    for cls in (0, 1):
        idx = np.where(lbl == cls)[0]
        keep_idx.append(idx[: args.per_class])
    keep_idx = np.concatenate(keep_idx)
    np.random.default_rng(args.seed).shuffle(keep_idx)
    wf, lbl = wf[keep_idx], lbl[keep_idx]
    print(f"  balanced: total={len(wf)}, Normal={int((lbl==0).sum())}, AFib={int((lbl==1).sum())}")

    if len(wf) < 200:
        raise RuntimeError("필터링 후 학습 데이터가 너무 적습니다. per_class/overdraw를 늘리세요.")

    # Train / Val 분리 (stratified 80:20)
    all_idx = np.arange(len(wf))
    train_idx, val_idx = train_test_split(
        all_idx, test_size=args.val_ratio, stratify=lbl, random_state=args.seed
    )
    ds_full = InMemoryECGDataset(wf, lbl)
    ds_train = Subset(ds_full, train_idx)
    ds_val = Subset(ds_full, val_idx)
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, drop_last=False)

    # 모델 / 옵티마이저 / 손실
    model = ECGEncoder(in_channels=3, embed_dim=128, dropout=args.dropout).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.5, patience=args.patience // 2, min_lr=1e-6
    )
    criterion = SupConLoss(temperature=args.temperature)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] ECGEncoder params={n_params:,}")

    # Early stopping 상태
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    # 학습 루프
    print(f"[train] epochs={args.epochs}, batch_size={args.batch_size}, "
          f"lr={args.lr}, T={args.temperature}, patience={args.patience}, "
          f"dropout={args.dropout}")
    stopped_epoch = args.epochs
    for epoch in range(1, args.epochs + 1):
        # --- train ---
        model.train()
        train_loss, n_batches = 0.0, 0
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

            train_loss += loss.item()
            n_batches += 1
        avg_train = train_loss / max(n_batches, 1)

        # --- val ---
        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                z = model(x)
                loss = criterion(z, y)
                val_loss += loss.item()
                n_val += 1
        avg_val = val_loss / max(n_val, 1)

        scheduler.step(avg_val)
        cur_lr = optim.param_groups[0]["lr"]
        print(f"  epoch {epoch:02d}/{args.epochs}  "
              f"train={avg_train:.4f}  val={avg_val:.4f}  "
              f"lr={cur_lr:.2e}  ({time.time()-t0:.1f}s)", end="")

        # --- early stopping ---
        if avg_val < best_val_loss - 1e-4:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            print("  *best*")
        else:
            no_improve += 1
            print(f"  (no improve {no_improve}/{args.patience})")
            if no_improve >= args.patience:
                print(f"[early stop] epoch {epoch} - patience {args.patience} 소진")
                stopped_epoch = epoch
                break

    # 최적 가중치 복원
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[restore] best val_loss={best_val_loss:.4f} 모델 복원")

    # 임베딩 추출 + t-SNE
    print("[eval] 임베딩 추출 및 t-SNE 시각화 ...")
    embs, labs = compute_embeddings(model, ds_full, device, batch_size=256)
    save_tsne(embs, labs, args.out,
              title=f"SupCon ECG embeddings (n={len(embs)}, T={args.temperature}, "
                    f"stopped={stopped_epoch}/{args.epochs})")

    # 가중치 저장 (옵션)
    if args.save_ckpt:
        ckpt_path = Path(args.out).with_suffix(".pt")
        torch.save({
            "model": model.state_dict(),
            "args": vars(args),
            "best_val_loss": best_val_loss,
            "stopped_epoch": stopped_epoch,
        }, ckpt_path)
        print(f"[save] ckpt → {ckpt_path}")

    print("[done]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="SNN/SupCon ECG 프로토타입 학습")
    p.add_argument("--h5", type=str, default=os.environ.get(
        "ECG_H5", r"E:\_archive\ecg_preprocessed.h5"))
    p.add_argument("--out", type=str, default="./snn_tsne_proto.png",
                   help="t-SNE PNG 저장 경로")
    p.add_argument("--per-class", type=int, default=5000)
    p.add_argument("--overdraw", type=float, default=1.5,
                   help="RR-CV 필터로 손실되는 분량을 감안한 초기 추출 배수")
    p.add_argument("--workers", type=int, default=6,
                   help="RR-CV 필터링 병렬 워커 수 (Ryzen 9700X 기준 6)")
    p.add_argument("--chunk-size", type=int, default=2000,
                   help="HDF5 청크 단위 (메모리 조절용, 2000=~480MB)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=10,
                   help="Early stopping patience (에폭 단위)")
    p.add_argument("--val-ratio", type=float, default=0.2,
                   help="Validation split 비율")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-ckpt", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)
