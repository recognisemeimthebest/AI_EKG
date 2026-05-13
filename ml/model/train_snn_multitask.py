"""
SNN/SupCon Multitask (4-class) 학습 + 평가.

4 클래스 (상호 배타적):
  0: Normal              — 부정맥 라벨=0, K<5.0
  1: AFib only           — 부정맥 라벨=1, K<5.0
  2: Other Arrhythmia    — 부정맥 라벨=2, K<5.0
  3: Hyperkalemia only   — 부정맥 라벨=0, K>=5.5

전략:
  - 클래스당 8,515개로 균형 (HK 클래스 기준)
  - 환자 단위 분할 (data leakage 방지)
  - SupCon Loss (multi-class 자동 지원)
  - 학습 후 multinomial LogisticRegression으로 4-class 분류
  - 분류기 .joblib 저장 (encoder.pt + classifier.joblib = 추론 파이프라인)

사용:
  python model/train_snn_multitask.py --epochs 100 --batch-size 256 \\
                                       --out g:/AIEKG/ml/checkpoints/snn-multitask-4cls
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import label_binarize

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))

from model.snn_encoder import ECGEncoder
from model.supcon_loss import SupConLoss


SAMPLING_RATE = 500
WINDOW_SAMPLES = 5000   # 10초

CLASS_NAMES = ["Normal", "AFib", "OtherArr", "HK"]


# ---------------------------------------------------------------------------
# 데이터셋 (in-memory)
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
# 4-class 인덱스 추출
# ---------------------------------------------------------------------------
def sample_4class(ecg_h5_path: str, hk_h5_path: str,
                  per_class: int = 8515,
                  k_normal_max: float = 5.0,
                  k_hk_min: float = 5.5,
                  time_diff_max_sec: int = 3600,
                  seed: int = 42):
    """
    4-class 균형 샘플링.

    Returns:
        ecg_indices: (N,) ecg_preprocessed.h5 인덱스
        labels:      (N,) 0/1/2/3
        subject_ids: (N,) 환자 단위 분할용
    """
    rng = np.random.default_rng(seed)

    # 부정맥 라벨 로드
    with h5py.File(ecg_h5_path, "r") as f:
        arr_label = f["label"][:]

    # HK 데이터 로드
    with h5py.File(hk_h5_path, "r") as f:
        hk_idx = f["indices"][:]
        k_val  = f["k_value"][:]
        td     = f["time_diff_sec"][:]
        sid    = f["subject_id"][:]

    # 시간차 필터
    tm = np.abs(td) < time_diff_max_sec
    hk_idx_t = hk_idx[tm]
    k_t      = k_val[tm]
    sid_t    = sid[tm]
    arr_t    = arr_label[hk_idx_t]

    # 4-class 마스크
    masks = {
        0: (arr_t == 0) & (k_t < k_normal_max),   # Normal
        1: (arr_t == 1) & (k_t < k_normal_max),   # AFib only
        2: (arr_t == 2) & (k_t < k_normal_max),   # Other Arr only
        3: (arr_t == 0) & (k_t >= k_hk_min),       # HK only
    }

    print("  원본 클래스별 가용 샘플:")
    for cls, m in masks.items():
        print(f"    Class {cls} {CLASS_NAMES[cls]:10s}: {m.sum():,}")

    # 각 클래스에서 per_class만큼 랜덤 추출
    sel_idx, sel_lbl, sel_sid = [], [], []
    for cls, m in masks.items():
        avail = np.where(m)[0]
        n_pick = min(per_class, len(avail))
        chosen = rng.choice(avail, size=n_pick, replace=False)
        sel_idx.append(hk_idx_t[chosen])
        sel_lbl.append(np.full(n_pick, cls, dtype=np.int64))
        sel_sid.append(sid_t[chosen])
        print(f"    pick Class {cls}: {n_pick:,}")

    sel_idx = np.concatenate(sel_idx)
    sel_lbl = np.concatenate(sel_lbl)
    sel_sid = np.concatenate(sel_sid)

    # 셔플 (같은 클래스 연속 방지)
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
# 청크 스트리밍 ECG 로드 + 3-lead 변환
# ---------------------------------------------------------------------------
def load_ecg_chunked(ecg_h5_path: str, sel_idx: np.ndarray, chunk_size: int = 2000):
    """선택된 인덱스의 waveform을 청크 단위로 읽고 3-lead (I, II, II-I) 변환."""
    n_total = len(sel_idx)
    t0 = time.time()
    # h5py 정렬 인덱스 접근 (랜덤 접근보다 ~3배 빠름)
    sort_order = np.argsort(sel_idx)
    sorted_idx = sel_idx[sort_order]
    inv_order = np.argsort(sort_order)
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
                print(f"    load {end:,}/{n_total:,}  ({time.time()-t0:.1f}s)")

    return sorted_out[inv_order]


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
        # train
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

        # val
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
    print("\n[probe] 임베딩 추출 (전체) ...")
    emb_all = extract_embeddings(model, wf_all, lbl_all, device, batch_size=256)

    print("[probe] LogisticRegression (multinomial) 학습 ...")
    # sklearn 1.5+ 에서 multi_class 인자 deprecated → 자동 결정 (>=3 class면 multinomial)
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

        # macro AUROC (one-vs-rest)
        y_bin = label_binarize(y, classes=[0, 1, 2, 3])
        auroc_macro = roc_auc_score(y_bin, prob, average="macro", multi_class="ovr")

        print(f"\n[{name}] n={len(y)}")
        print(f"  Acc={acc:.4f}  F1(macro)={f1_macro:.4f}  AUROC(macro)={auroc_macro:.4f}")
        print(classification_report(y, pred, target_names=CLASS_NAMES,
                                    digits=3, zero_division=0))
        print(f"  Confusion matrix:\n{confusion_matrix(y, pred)}")

        # 클래스별 one-vs-rest AUROC
        cls_auroc = {}
        for i, cname in enumerate(CLASS_NAMES):
            try:
                cls_auroc[cname] = roc_auc_score((y == i).astype(int), prob[:, i])
            except Exception:
                cls_auroc[cname] = float("nan")
        print(f"  Per-class AUROC (one-vs-rest): {cls_auroc}")

        results[name] = dict(
            acc=acc, f1_macro=f1_macro, auroc_macro=auroc_macro,
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

    # 1. 4-class 인덱스 추출
    print("\n[1/5] 4-class 데이터 추출 ...")
    sel_idx, lbl, sid = sample_4class(
        args.ecg_h5, args.hk_h5,
        per_class=args.per_class,
        k_normal_max=args.k_normal_max,
        k_hk_min=args.k_hk_min,
        time_diff_max_sec=args.time_diff_max,
        seed=args.seed,
    )
    print(f"  총 {len(sel_idx):,} 샘플, 클래스 분포: {np.bincount(lbl)}")

    # 2. 환자 단위 분할
    print("\n[2/5] 환자 단위 분할 (70/15/15) ...")
    tr_mask, va_mask, te_mask = patient_level_split(sid, seed=args.seed)
    print(f"  Train: {tr_mask.sum():,}, Val: {va_mask.sum():,}, Test: {te_mask.sum():,}")
    for cls in range(4):
        for name, m in [("tr", tr_mask), ("va", va_mask), ("te", te_mask)]:
            cnt = ((lbl == cls) & m).sum()
            print(f"    Class {cls} {CLASS_NAMES[cls]:10s} {name}: {cnt:,}", end="  ")
        print()

    # 3. ECG 로드 (한 번에)
    print(f"\n[3/5] ECG 로드 (chunk={args.chunk_size}) ...")
    wf_all = load_ecg_chunked(args.ecg_h5, sel_idx, chunk_size=args.chunk_size)
    print(f"  shape: {wf_all.shape}, dtype={wf_all.dtype}, RAM={wf_all.nbytes/1024**2:.0f} MB")

    # 4. 학습
    print(f"\n[4/5] SNN 학습 ...")
    ds_tr = InMemoryECGDataset(wf_all[tr_mask], lbl[tr_mask])
    ds_va = InMemoryECGDataset(wf_all[va_mask], lbl[va_mask])
    train_loader = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader   = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                              num_workers=0, drop_last=False)
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = ECGEncoder(in_channels=3, embed_dim=128, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  ECGEncoder params={n_params:,}")

    enc_path = out_dir / "encoder.pt"
    if args.skip_train and enc_path.exists():
        print(f"[skip-train] 기존 encoder 로드: {enc_path}")
        ckpt = torch.load(enc_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        best_val = ckpt.get("best_val_loss", 0.0)
        stopped_epoch = ckpt.get("stopped_epoch", 0)
    else:
        best_val, stopped_epoch = train_supcon(model, train_loader, val_loader, device, args)

    # 인코더 저장 (학습 했을 때만)
    if not (args.skip_train and enc_path.exists()):
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
        model, wf_all, lbl, tr_mask, va_mask, te_mask, device, args.seed
    )

    # 분류기 저장
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

    # 결과 json
    import json
    json_path = out_dir / "train_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "best_val_loss": best_val,
            "stopped_epoch": stopped_epoch,
            "n_params": n_params,
            "class_names": CLASS_NAMES,
            "results": {k: {kk: (vv if not isinstance(vv, dict) else vv)
                             for kk, vv in v.items()} for k, v in results.items()},
            "args": vars(args),
        }, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
    print(f"[save] results → {json_path}")
    print("\n[done]")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ecg-h5", type=str,
                   default=os.environ.get("ECG_H5", r"E:\_archive\ecg_preprocessed.h5"))
    p.add_argument("--hk-h5", type=str,
                   default="G:/AIEKG/ml/data/ecg_hyperkalemia_v2.h5")
    p.add_argument("--out", type=str,
                   default="g:/AIEKG/ml/checkpoints/snn-multitask-4cls")
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
    p.add_argument("--skip-train",     action="store_true",
                   help="기존 encoder.pt가 있으면 학습 스킵하고 평가만 실행")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
