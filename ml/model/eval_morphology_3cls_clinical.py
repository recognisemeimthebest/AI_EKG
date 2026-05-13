"""
3-class morphology + 임상변수 (나이, 고혈압, 당뇨) 평가.

비교:
  A. 임베딩만 (128-dim)
  B. 임베딩 + 나이 (129)
  C. 임베딩 + 나이 + HTN + DM (131)
"""
import argparse
import os
import sys
import json
from pathlib import Path

import h5py
import numpy as np
import torch
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
    MorphologyEncoder, patient_level_split, load_ecg_split_chunked,
    extract_embeddings,
)
from model.train_morphology_3cls import sample_3class_morphology, CLASS_NAMES


def evaluate_classifier(clf, X, y, name):
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
    return dict(acc=float(acc), f1_macro=float(f1_macro),
                auroc_macro=float(auroc_macro),
                per_class_auroc=cls_auroc)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # 1. 3-class 데이터
    print("\n[1/5] 3-class 인덱스 추출 ...")
    sel_idx, lbl, sid = sample_3class_morphology(
        args.ecg_h5, args.hk_h5,
        per_class=args.per_class,
        k_normal_max=args.k_normal_max,
        time_diff_max_sec=args.time_diff_max,
        seed=args.seed,
    )
    print(f"  총 {len(sel_idx):,} ECG")

    # 2. 환자 분할
    print("\n[2/5] 환자 분할 ...")
    tr_mask, va_mask, te_mask = patient_level_split(sid, seed=args.seed)
    print(f"  Train: {tr_mask.sum():,}, Val: {va_mask.sum():,}, Test: {te_mask.sum():,}")

    # 3. 임상 변수 추출
    print("\n[3/5] 임상 변수 추출 ...")
    with h5py.File(args.ecg_h5, "r") as f:
        all_pat = f["patient_features"][:]
        # patient_feature_names = ['age', 'gender_code']
        ages = all_pat[sel_idx, 0].astype(np.float32)
    print(f"  age: min={ages.min():.1f}, max={ages.max():.1f}, mean={ages.mean():.1f}")

    # clinical_features.h5 ─ subject_id → (dm, hf, mi, aht)
    with h5py.File(args.clinical_h5, "r") as f:
        clin_sid = f["subject_id"][:]
        clin_flags = f["clinical_flags"][:]
        names = list(f.attrs["feature_names"])
    print(f"  clinical: subjects={len(clin_sid):,}, flags={names}")

    # subject_id 매핑
    sid_to_idx = {int(s): i for i, s in enumerate(clin_sid)}
    htn_arr = np.zeros(len(sel_idx), dtype=np.float32)
    dm_arr  = np.zeros(len(sel_idx), dtype=np.float32)
    missing = 0
    for i, s in enumerate(sid):
        j = sid_to_idx.get(int(s))
        if j is not None:
            dm_arr[i]  = clin_flags[j, 0]   # DM
            htn_arr[i] = clin_flags[j, 3]   # AHT
        else:
            missing += 1
    print(f"  매칭: 성공 {len(sid)-missing:,} / 누락 {missing:,}")
    print(f"  HTN 양성률: {htn_arr.mean():.3%} | DM 양성률: {dm_arr.mean():.3%}")

    # 4. ECG 로드 + 5초 split
    print(f"\n[4/5] ECG 로드 + 5초 split ...")
    wf_2x, lbl_2x, (tr_2x, va_2x, te_2x) = load_ecg_split_chunked(
        args.ecg_h5, sel_idx, lbl, sid,
        split_masks=(tr_mask, va_mask, te_mask),
        chunk_size=args.chunk_size,
    )
    # 임상 변수도 2배
    age_2x = np.concatenate([ages, ages], axis=0)
    htn_2x = np.concatenate([htn_arr, htn_arr], axis=0)
    dm_2x  = np.concatenate([dm_arr, dm_arr], axis=0)

    # 5. encoder 로드 + 임베딩 추출
    print(f"\n[5/5] encoder 로드 + 임베딩 ...")
    model = MorphologyEncoder(in_channels=3, embed_dim=128, dropout=0.2).to(device)
    ckpt = torch.load(args.encoder, map_location=device)
    model.load_state_dict(ckpt["model"])
    emb_all = extract_embeddings(model, wf_2x, lbl_2x, device, batch_size=256)
    print(f"  emb shape: {emb_all.shape}")

    # ============================================================
    # 3가지 비교
    # ============================================================
    print("\n" + "=" * 70)
    print("[비교] 3-class 임베딩 vs +나이 vs +나이+HTN+DM")
    print("=" * 70)

    results_all = {}

    # age 정규화
    age_mean = age_2x[tr_2x].mean(); age_std = age_2x[tr_2x].std()
    age_z = (age_2x - age_mean) / age_std

    # 입력 구성
    X_a = emb_all
    X_b = np.concatenate([emb_all, age_z.reshape(-1, 1)], axis=1)
    X_c = np.concatenate([emb_all, age_z.reshape(-1, 1),
                          htn_2x.reshape(-1, 1), dm_2x.reshape(-1, 1)], axis=1)

    for tag, X in [("A_emb_only", X_a),
                   ("B_emb_age", X_b),
                   ("C_emb_age_htn_dm", X_c)]:
        print(f"\n=== {tag} ({X.shape[1]}-dim) ===")
        clf = LogisticRegression(max_iter=2000, random_state=args.seed,
                                  C=1.0, class_weight="balanced", solver="lbfgs")
        clf.fit(X[tr_2x], lbl_2x[tr_2x])
        results_all[tag] = {
            "Val":  evaluate_classifier(clf, X[va_2x], lbl_2x[va_2x], f"Val  ({tag})"),
            "Test": evaluate_classifier(clf, X[te_2x], lbl_2x[te_2x], f"Test ({tag})"),
        }
        if tag == "C_emb_age_htn_dm":
            # coefficient 확인 — age/HTN/DM 마지막 3개
            print(f"  마지막 3 변수 가중치 (class별):")
            for cls_idx, cname in enumerate(CLASS_NAMES):
                w = clf.coef_[cls_idx, -3:]
                print(f"    {cname:10s}: age={w[0]:+.3f}, HTN={w[1]:+.3f}, DM={w[2]:+.3f}")

    # ===== 비교 summary =====
    print("\n" + "=" * 70)
    print("[SUMMARY] Test 결과 비교 (3-class)")
    print("=" * 70)
    a = results_all["A_emb_only"]["Test"]
    b = results_all["B_emb_age"]["Test"]
    c = results_all["C_emb_age_htn_dm"]["Test"]
    print(f"\n{'Metric':22s} | A: emb | B: +age  | C: +age+HTN+DM | C-A")
    print(f"{'Acc':22s} | {a['acc']:.4f} | {b['acc']:.4f}   | {c['acc']:.4f}        | {c['acc']-a['acc']:+.4f}")
    print(f"{'F1 (macro)':22s} | {a['f1_macro']:.4f} | {b['f1_macro']:.4f}   | {c['f1_macro']:.4f}        | {c['f1_macro']-a['f1_macro']:+.4f}")
    print(f"{'AUROC (macro)':22s} | {a['auroc_macro']:.4f} | {b['auroc_macro']:.4f}   | {c['auroc_macro']:.4f}        | {c['auroc_macro']-a['auroc_macro']:+.4f}")
    print(f"\n{'Class':10s} | A AUROC | B AUROC | C AUROC | C-A")
    for cname in CLASS_NAMES:
        aa = a['per_class_auroc'][cname]; bb = b['per_class_auroc'][cname]; cc = c['per_class_auroc'][cname]
        print(f"{cname:10s} | {aa:.4f}  | {bb:.4f}  | {cc:.4f}  | {cc-aa:+.4f}")

    # 저장
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "eval_clinical.json", "w") as f:
        json.dump(results_all, f, indent=2)
    print(f"\n[save] → {out_path / 'eval_clinical.json'}")
    print("[done]")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ecg-h5", type=str,
                   default=os.environ.get("ECG_H5", r"E:\_archive\ecg_preprocessed.h5"))
    p.add_argument("--hk-h5", type=str,
                   default="G:/AIEKG/ml/data/ecg_hyperkalemia_v2.h5")
    p.add_argument("--clinical-h5", type=str,
                   default="G:/AIEKG/ml/data/clinical_features.h5")
    p.add_argument("--encoder", type=str,
                   default="g:/AIEKG/ml/checkpoints/morphology-3cls/encoder.pt")
    p.add_argument("--out", type=str,
                   default="g:/AIEKG/ml/checkpoints/morphology-3cls")
    p.add_argument("--per-class",      type=int, default=8515)
    p.add_argument("--k-normal-max",   type=float, default=5.0)
    p.add_argument("--time-diff-max",  type=int, default=3600)
    p.add_argument("--chunk-size",     type=int, default=2000)
    p.add_argument("--seed",           type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
