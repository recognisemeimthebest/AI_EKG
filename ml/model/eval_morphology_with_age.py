"""
Morphology 4-class encoder + 나이(age) 변수 추가 평가.

기존 학습된 encoder.pt는 그대로 사용하고, LogisticRegression 분류기만 다시 학습.
입력: 128-dim ECG 임베딩 + 1-dim age(정규화) = 129-dim

비교:
  - 임베딩만: 이전 결과 (AUROC macro 0.906)
  - 임베딩+나이: 이번 결과
"""
import argparse
import os
import sys
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import label_binarize, StandardScaler

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))

from model.train_morphology_4cls import (
    MorphologyEncoder, sample_4class_morphology, patient_level_split,
    load_ecg_split_chunked, extract_embeddings, InMemoryECGDataset, CLASS_NAMES
)


def evaluate_classifier(clf, X, y, name, scaler=None):
    if scaler is not None:
        X = scaler.transform(X)
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
    print(f"  Confusion:\n{confusion_matrix(y, pred)}")
    cls_auroc = {}
    for i, cname in enumerate(CLASS_NAMES):
        try:
            cls_auroc[cname] = float(roc_auc_score((y == i).astype(int), prob[:, i]))
        except Exception:
            cls_auroc[cname] = float("nan")
    print(f"  Per-class AUROC: {cls_auroc}")
    return dict(acc=float(acc), f1_macro=float(f1_macro),
                auroc_macro=float(auroc_macro),
                per_class_auroc=cls_auroc)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # 1. 데이터 추출 (동일 procedure)
    print("\n[1/4] 4-class 인덱스 추출 ...")
    sel_idx, lbl, sid = sample_4class_morphology(
        args.ecg_h5, args.hk_h5,
        per_class=args.per_class,
        k_normal_max=args.k_normal_max,
        k_hk_min=args.k_hk_min,
        time_diff_max_sec=args.time_diff_max,
        seed=args.seed,
    )
    print(f"  총 {len(sel_idx):,} ECG")

    # 2. 환자 분할
    print("\n[2/4] 환자 단위 분할 ...")
    tr_mask, va_mask, te_mask = patient_level_split(sid, seed=args.seed)
    print(f"  Train: {tr_mask.sum():,}, Val: {va_mask.sum():,}, Test: {te_mask.sum():,}")

    # 3. ECG + 나이 로드, 5초 split
    print(f"\n[3/4] ECG 로드 + age 추출 + 5초 split ...")

    # 나이 추출 (patient_features의 0번 = age)
    with h5py.File(args.ecg_h5, "r") as f:
        all_pat = f["patient_features"][:]
        # patient_feature_names = ['age', 'gender_code']
        ages = all_pat[sel_idx, 0].astype(np.float32)
    print(f"  age stats: min={ages.min():.1f}  max={ages.max():.1f}  mean={ages.mean():.1f}")

    # ECG split (앞/뒤 5초)
    wf_2x, lbl_2x, (tr_2x, va_2x, te_2x) = load_ecg_split_chunked(
        args.ecg_h5, sel_idx, lbl, sid,
        split_masks=(tr_mask, va_mask, te_mask),
        chunk_size=args.chunk_size,
    )
    # age도 2배 (앞/뒤 동일 환자라 같음)
    age_2x = np.concatenate([ages, ages], axis=0)

    # 4. encoder 로드 + 임베딩 추출
    print(f"\n[4/4] encoder 로드 + 임베딩 추출 ...")
    model = MorphologyEncoder(in_channels=3, embed_dim=128, dropout=0.2).to(device)
    ckpt = torch.load(args.encoder, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"  encoder loaded: {args.encoder}")

    emb_all = extract_embeddings(model, wf_2x, lbl_2x, device, batch_size=256)
    print(f"  emb shape: {emb_all.shape}")

    # ---- 5. 분류기 학습 (3가지 비교) ----
    print("\n" + "=" * 70)
    print("[비교] 임베딩만 vs 임베딩+나이")
    print("=" * 70)

    results_all = {}

    # ===== A. 임베딩만 (기존 결과 재현) =====
    print("\n=== A. 임베딩만 (128-dim) ===")
    clf_a = LogisticRegression(max_iter=2000, random_state=args.seed,
                                C=1.0, class_weight="balanced", solver="lbfgs")
    clf_a.fit(emb_all[tr_2x], lbl_2x[tr_2x])
    results_all["embedding_only"] = {
        "Val":  evaluate_classifier(clf_a, emb_all[va_2x], lbl_2x[va_2x], "Val (emb)"),
        "Test": evaluate_classifier(clf_a, emb_all[te_2x], lbl_2x[te_2x], "Test (emb)"),
    }

    # ===== B. 임베딩 + 나이 =====
    print("\n=== B. 임베딩(128) + 나이(1) = 129-dim ===")
    age_col = age_2x.reshape(-1, 1)
    X_all = np.concatenate([emb_all, age_col], axis=1)

    # age 정규화 (z-score, train 기준)
    scaler = StandardScaler()
    X_all_scaled = X_all.copy()
    age_train = age_2x[tr_2x].reshape(-1, 1)
    age_mean, age_std = age_train.mean(), age_train.std()
    X_all_scaled[:, -1] = (X_all[:, -1] - age_mean) / age_std
    print(f"  age 정규화: mean={age_mean:.2f}, std={age_std:.2f}")

    clf_b = LogisticRegression(max_iter=2000, random_state=args.seed,
                                C=1.0, class_weight="balanced", solver="lbfgs")
    clf_b.fit(X_all_scaled[tr_2x], lbl_2x[tr_2x])
    results_all["embedding_plus_age"] = {
        "Val":  evaluate_classifier(clf_b, X_all_scaled[va_2x], lbl_2x[va_2x], "Val (emb+age)"),
        "Test": evaluate_classifier(clf_b, X_all_scaled[te_2x], lbl_2x[te_2x], "Test (emb+age)"),
    }

    # ===== 비교 summary =====
    print("\n" + "=" * 70)
    print("[SUMMARY] Test 결과 비교")
    print("=" * 70)
    a = results_all["embedding_only"]["Test"]
    b = results_all["embedding_plus_age"]["Test"]
    print(f"\n  Metric               | A: emb only | B: emb+age | delta")
    print(f"  Acc                  | {a['acc']:.4f}      | {b['acc']:.4f}     | {b['acc']-a['acc']:+.4f}")
    print(f"  F1 (macro)           | {a['f1_macro']:.4f}      | {b['f1_macro']:.4f}     | {b['f1_macro']-a['f1_macro']:+.4f}")
    print(f"  AUROC (macro)        | {a['auroc_macro']:.4f}      | {b['auroc_macro']:.4f}     | {b['auroc_macro']-a['auroc_macro']:+.4f}")
    print(f"\n  Class    | A AUROC | B AUROC | delta")
    for cname in CLASS_NAMES:
        aa = a['per_class_auroc'][cname]; bb = b['per_class_auroc'][cname]
        print(f"  {cname:8s} | {aa:.4f}  | {bb:.4f}  | {bb-aa:+.4f}")

    # 저장
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "eval_with_age.json", "w") as f:
        json.dump(results_all, f, indent=2)
    import joblib
    joblib.dump({
        "classifier_with_age": clf_b,
        "age_mean": float(age_mean), "age_std": float(age_std),
        "class_names": CLASS_NAMES,
        "encoder_ckpt": args.encoder,
    }, out_path / "classifier_with_age.joblib")
    print(f"\n[save] → {out_path}")
    print("[done]")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ecg-h5", type=str,
                   default=os.environ.get("ECG_H5", r"E:\_archive\ecg_preprocessed.h5"))
    p.add_argument("--hk-h5", type=str,
                   default="G:/AIEKG/ml/data/ecg_hyperkalemia_v2.h5")
    p.add_argument("--encoder", type=str,
                   default="g:/AIEKG/ml/checkpoints/morphology-4cls/encoder.pt")
    p.add_argument("--out", type=str,
                   default="g:/AIEKG/ml/checkpoints/morphology-4cls")
    p.add_argument("--per-class",      type=int, default=8515)
    p.add_argument("--k-normal-max",   type=float, default=5.0)
    p.add_argument("--k-hk-min",       type=float, default=5.5)
    p.add_argument("--time-diff-max",  type=int, default=3600)
    p.add_argument("--chunk-size",     type=int, default=2000)
    p.add_argument("--seed",           type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
