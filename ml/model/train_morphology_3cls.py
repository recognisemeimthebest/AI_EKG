"""
Morphology-focused 3-class ECG classifier (Normal/AFib/OtherArr).
HK 제외 → Normal 클래스의 혼동이 줄어드는지 확인.

train_morphology_4cls.py 와 동일한 구조, 단 HK 클래스 제거.
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

from model.train_morphology_4cls import (
    MorphologyEncoder, InMemoryECGDataset,
    patient_level_split, load_ecg_split_chunked,
    extract_embeddings, train_supcon,
)
from model.supcon_loss import SupConLoss


CLASS_NAMES = ["Normal", "AFib", "OtherArr"]
WINDOW_FULL = 5000
WINDOW_HALF = 2500


def sample_3class_morphology(ecg_h5_path, hk_h5_path,
                              per_class=8515, k_normal_max=5.0,
                              time_diff_max_sec=3600, seed=42):
    """
    3-class 균형 샘플링 (HK 제외, K<5.0 정상 K값 가진 ECG만):
      0: Normal     — label==0, K<5.0
      1: AFib       — label==1, K<5.0
      2: Other Arr  — label==2, K<5.0
    """
    rng = np.random.default_rng(seed)

    print("  로드: ecg_preprocessed.h5 label ...")
    with h5py.File(ecg_h5_path, "r") as f:
        arr_label = f["label"][:]
        all_subjects = f["subject_id"][:]

    print("  로드: ecg_hyperkalemia_v2.h5 (K<{} 필터용) ...".format(k_normal_max))
    with h5py.File(hk_h5_path, "r") as f:
        hk_indices = f["indices"][:]
        k_values   = f["k_value"][:]
        time_diffs = f["time_diff_sec"][:]

    tm = np.abs(time_diffs) < time_diff_max_sec
    hk_idx_filt = hk_indices[tm]
    k_filt      = k_values[tm]
    k_lookup = np.full(len(arr_label), np.nan, dtype=np.float32)
    k_lookup[hk_idx_filt] = k_filt

    has_k = ~np.isnan(k_lookup)
    m0 = (arr_label == 0) & has_k & (k_lookup < k_normal_max)
    m1 = (arr_label == 1) & has_k & (k_lookup < k_normal_max)
    m2 = (arr_label == 2) & has_k & (k_lookup < k_normal_max)

    print("  원본 클래스별 가용 샘플:")
    for cls, m in enumerate([m0, m1, m2]):
        print(f"    Class {cls} {CLASS_NAMES[cls]:10s}: {m.sum():,}")

    sel_idx, sel_lbl, sel_sid = [], [], []
    for cls, m in enumerate([m0, m1, m2]):
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


def linear_probe_eval_3cls(model, wf_all, lbl_all, tr_mask, va_mask, te_mask, device, seed):
    print("\n[probe] 임베딩 추출 ...")
    emb_all = extract_embeddings(model, wf_all, lbl_all, device, batch_size=256)

    print("[probe] LogisticRegression (3-class) 학습 ...")
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

        y_bin = label_binarize(y, classes=[0, 1, 2])
        auroc_macro = roc_auc_score(y_bin, prob, average="macro", multi_class="ovr")

        print(f"\n[{name}] n={len(y)}")
        print(f"  Acc={acc:.4f}  F1(macro)={f1_macro:.4f}  AUROC(macro)={auroc_macro:.4f}")
        print(classification_report(y, pred, target_names=CLASS_NAMES,
                                    digits=3, zero_division=0))
        print(f"  Confusion:\n{confusion_matrix(y, pred)}")

        cls_auroc = {}
        for i, cname in enumerate(CLASS_NAMES):
            try:
                cls_auroc[cname] = float(roc_auc_score((y == i).astype(int), prob[:, i]))
            except Exception:
                cls_auroc[cname] = float("nan")
        print(f"  Per-class AUROC: {cls_auroc}")

        results[name] = dict(acc=float(acc), f1_macro=float(f1_macro),
                             auroc_macro=float(auroc_macro),
                             per_class_auroc=cls_auroc)

    return clf, results, emb_all


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    if device.type == "cuda":
        print(f"[gpu]    {torch.cuda.get_device_name(0)}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 3-class 인덱스 추출
    print("\n[1/5] 3-class 인덱스 추출 (HK 제외) ...")
    sel_idx, lbl, sid = sample_3class_morphology(
        args.ecg_h5, args.hk_h5,
        per_class=args.per_class,
        k_normal_max=args.k_normal_max,
        time_diff_max_sec=args.time_diff_max,
        seed=args.seed,
    )
    print(f"  총 {len(sel_idx):,} ECG, 클래스 분포: {np.bincount(lbl)}")

    # 2. 환자 분할
    print("\n[2/5] 환자 단위 분할 ...")
    tr_mask, va_mask, te_mask = patient_level_split(sid, seed=args.seed)
    print(f"  Train: {tr_mask.sum():,}, Val: {va_mask.sum():,}, Test: {te_mask.sum():,}")

    # 3. 5초 split
    print(f"\n[3/5] ECG 로드 + 5초 split ...")
    wf_2x, lbl_2x, (tr_2x, va_2x, te_2x) = load_ecg_split_chunked(
        args.ecg_h5, sel_idx, lbl, sid,
        split_masks=(tr_mask, va_mask, te_mask),
        chunk_size=args.chunk_size,
    )
    print(f"  최종: shape={wf_2x.shape}, RAM={wf_2x.nbytes/1024**2:.0f} MB")

    # 4. 학습
    print(f"\n[4/5] CNN-TCN + SupCon 학습 (3-class) ...")
    ds_tr = InMemoryECGDataset(wf_2x[tr_2x], lbl_2x[tr_2x])
    ds_va = InMemoryECGDataset(wf_2x[va_2x], lbl_2x[va_2x])
    train_loader = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader   = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                              num_workers=0, drop_last=False)

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

    # 5. 평가
    print(f"\n[5/5] Linear probe 평가 (3-class) ...")
    clf, results, _ = linear_probe_eval_3cls(
        model, wf_2x, lbl_2x, tr_2x, va_2x, te_2x, device, args.seed
    )

    import joblib
    joblib.dump({
        "classifier": clf, "class_names": CLASS_NAMES,
        "encoder_ckpt": str(enc_path), "results": results,
    }, out_dir / "classifier.joblib")

    with open(out_dir / "train_results.json", "w") as f:
        json.dump({
            "best_val_loss": float(best_val),
            "stopped_epoch": int(stopped_epoch),
            "n_params": int(n_params),
            "class_names": CLASS_NAMES,
            "results": results,
            "args": vars(args),
        }, f, indent=2)
    print(f"\n[save] → {out_dir}")
    print("[done]")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ecg-h5", type=str,
                   default=os.environ.get("ECG_H5", r"E:\_archive\ecg_preprocessed.h5"))
    p.add_argument("--hk-h5", type=str,
                   default="G:/AIEKG/ml/data/ecg_hyperkalemia_v2.h5")
    p.add_argument("--out", type=str,
                   default="g:/AIEKG/ml/checkpoints/morphology-3cls")
    p.add_argument("--per-class",      type=int, default=8515)
    p.add_argument("--k-normal-max",   type=float, default=5.0)
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
