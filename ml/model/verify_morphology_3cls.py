"""
Morphology 3-class 모델 신뢰도 검증.

체크 항목:
  1. 환자 단위 분할 정합성 (train/val/test에 같은 환자 X)
  2. 5초 split의 같은 ECG가 다른 split으로 새지 않았는지
  3. 클래스 균형 (train/val/test 각각)
  4. Confusion matrix 재계산 (saved encoder + LR로)
  5. K값 분포 (학습/평가 세트별)
  6. Subject 다양성 (환자 수, ECG 수)
  7. Class별 prediction probability 분포 (overconfident인가?)
  8. AUROC 신뢰구간 (bootstrap)
"""
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, accuracy_score, classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))

from model.train_morphology_4cls import (
    MorphologyEncoder, patient_level_split, load_ecg_split_chunked,
    extract_embeddings,
)
from model.train_morphology_3cls import sample_3class_morphology, CLASS_NAMES


SEED = 42
ECG_H5 = r"E:\_archive\ecg_preprocessed.h5"
HK_H5 = "G:/AIEKG/ml/data/ecg_hyperkalemia_v2.h5"
ENC = "g:/AIEKG/ml/checkpoints/morphology-3cls/encoder.pt"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    print("\n" + "=" * 70)
    print("[1] 데이터 추출 + 분할 검증")
    print("=" * 70)
    sel_idx, lbl, sid = sample_3class_morphology(ECG_H5, HK_H5, per_class=8515,
                                                  k_normal_max=5.0,
                                                  time_diff_max_sec=3600,
                                                  seed=SEED)
    tr_mask, va_mask, te_mask = patient_level_split(sid, seed=SEED)

    # 1-1. 환자 누수 검증
    tr_pat = set(sid[tr_mask].tolist())
    va_pat = set(sid[va_mask].tolist())
    te_pat = set(sid[te_mask].tolist())
    overlap_tv = tr_pat & va_pat
    overlap_tt = tr_pat & te_pat
    overlap_vt = va_pat & te_pat
    print(f"\n  환자 단위 분할:")
    print(f"    Train 환자: {len(tr_pat):,}")
    print(f"    Val 환자:   {len(va_pat):,}")
    print(f"    Test 환자:  {len(te_pat):,}")
    print(f"    Train ∩ Val:  {len(overlap_tv)} (0이어야 함)")
    print(f"    Train ∩ Test: {len(overlap_tt)} (0이어야 함)")
    print(f"    Val ∩ Test:   {len(overlap_vt)} (0이어야 함)")
    if overlap_tv or overlap_tt or overlap_vt:
        print(f"  [FAIL] 데이터 누수 발견!")
    else:
        print(f"  [OK] 환자 누수 없음")

    # 1-2. 클래스 균형
    print(f"\n  클래스 분포:")
    for cls in range(3):
        tr_n = ((lbl == cls) & tr_mask).sum()
        va_n = ((lbl == cls) & va_mask).sum()
        te_n = ((lbl == cls) & te_mask).sum()
        print(f"    Class {cls} {CLASS_NAMES[cls]:10s}  tr={tr_n:5d}  va={va_n:5d}  te={te_n:5d}")

    # 1-3. K값 분포 확인
    print("\n  K값 분포 (학습 데이터 = K<5.0):")
    with h5py.File(HK_H5, "r") as f:
        hk_idx_all = f["indices"][:]
        k_all = f["k_value"][:]
        td_all = f["time_diff_sec"][:]
    tm = np.abs(td_all) < 3600
    k_lookup = np.full(800000, np.nan, dtype=np.float32)
    k_lookup[hk_idx_all[tm]] = k_all[tm]
    k_used = k_lookup[sel_idx]
    print(f"    K 평균: {np.nanmean(k_used):.2f}, std: {np.nanstd(k_used):.2f}")
    print(f"    K 분포: 4.0이하 {np.sum(k_used<4.0):,}, 4.0~4.5 {np.sum((k_used>=4.0)&(k_used<4.5)):,}, "
          f"4.5~5.0 {np.sum((k_used>=4.5)&(k_used<5.0)):,}")

    # 1-4. ECG 다양성
    print(f"\n  ECG / 환자 비율:")
    n_ecg = len(sel_idx); n_pat = len(set(sid.tolist()))
    print(f"    총 ECG: {n_ecg:,}, 총 환자: {n_pat:,}, 환자당 평균 ECG: {n_ecg/n_pat:.2f}")

    # 2. ECG 로드 + 임베딩 + 평가 재현
    print("\n" + "=" * 70)
    print("[2] 결과 재현 + 다른 seed 검증")
    print("=" * 70)
    print("  ECG 로드 + 5초 split (~5분)...")
    wf_2x, lbl_2x, (tr_2x, va_2x, te_2x) = load_ecg_split_chunked(
        ECG_H5, sel_idx, lbl, sid,
        split_masks=(tr_mask, va_mask, te_mask), chunk_size=2000,
    )

    # 5초 split 누수 검증: 같은 ECG의 앞/뒤 5초가 다른 split으로 새는가?
    n_orig = len(sel_idx)
    front_in_tr = tr_2x[:n_orig]
    back_in_tr  = tr_2x[n_orig:]
    front_in_te = te_2x[:n_orig]
    back_in_te  = te_2x[n_orig:]
    leak_count = ((front_in_tr & back_in_te) | (back_in_tr & front_in_te)).sum()
    print(f"\n  5초 split 누수 검증:")
    print(f"    같은 ECG의 앞/뒤가 다른 split에 있는 건수: {leak_count} (0이어야 함)")
    if leak_count == 0:
        print(f"    [OK] 5초 split 누수 없음")

    # 임베딩 추출
    print("\n  encoder 로드 + 임베딩 추출...")
    model = MorphologyEncoder(in_channels=3, embed_dim=128, dropout=0.2).to(device)
    ckpt = torch.load(ENC, map_location=device)
    model.load_state_dict(ckpt["model"])
    emb = extract_embeddings(model, wf_2x, lbl_2x, device, batch_size=256)

    # LR 학습 + Test 평가
    clf = LogisticRegression(max_iter=2000, random_state=SEED, C=1.0,
                              class_weight="balanced", solver="lbfgs")
    clf.fit(emb[tr_2x], lbl_2x[tr_2x])

    # 3. Test 결과 + bootstrap CI
    print("\n" + "=" * 70)
    print("[3] Test 평가 + Bootstrap 95% CI (AUROC)")
    print("=" * 70)
    X_te = emb[te_2x]; y_te = lbl_2x[te_2x]
    pred = clf.predict(X_te)
    prob = clf.predict_proba(X_te)
    acc = accuracy_score(y_te, pred)
    y_bin = label_binarize(y_te, classes=[0, 1, 2])
    auroc_macro = roc_auc_score(y_bin, prob, average="macro", multi_class="ovr")
    print(f"\n  Test Acc: {acc:.4f}, AUROC macro: {auroc_macro:.4f}")
    print(f"\n  Confusion matrix:\n{confusion_matrix(y_te, pred)}")
    print(classification_report(y_te, pred, target_names=CLASS_NAMES, digits=3))

    # Bootstrap CI
    print("\n  Bootstrap 95% CI (n=200 resamples):")
    rng = np.random.default_rng(SEED)
    n = len(y_te)
    aurocs_macro = []
    aurocs_per_class = {c: [] for c in CLASS_NAMES}
    for _ in range(200):
        idx = rng.choice(n, n, replace=True)
        try:
            yb = label_binarize(y_te[idx], classes=[0, 1, 2])
            aurocs_macro.append(roc_auc_score(yb, prob[idx], average="macro", multi_class="ovr"))
            for i, cname in enumerate(CLASS_NAMES):
                aurocs_per_class[cname].append(
                    roc_auc_score((y_te[idx] == i).astype(int), prob[idx, i])
                )
        except Exception:
            pass
    am = np.array(aurocs_macro)
    print(f"    AUROC macro: {am.mean():.4f} [95% CI: {np.percentile(am,2.5):.4f}, {np.percentile(am,97.5):.4f}]")
    for cname in CLASS_NAMES:
        a = np.array(aurocs_per_class[cname])
        print(f"    {cname:10s}: {a.mean():.4f} [95% CI: {np.percentile(a,2.5):.4f}, {np.percentile(a,97.5):.4f}]")

    # 4. Probability calibration 체크 (overconfident인가?)
    print("\n" + "=" * 70)
    print("[4] Prediction probability 분포 (overconfident 진단)")
    print("=" * 70)
    max_probs = prob.max(axis=1)
    print(f"  max prob 분포:")
    print(f"    >0.95 (매우 확신): {(max_probs > 0.95).mean():.1%}")
    print(f"    >0.80          : {(max_probs > 0.80).mean():.1%}")
    print(f"    >0.50          : {(max_probs > 0.50).mean():.1%}")
    print(f"    mean           : {max_probs.mean():.4f}")

    # Calibration: 0.9 이상으로 확신한 예측의 실제 정답률
    high_conf = max_probs > 0.9
    if high_conf.any():
        acc_high = (pred[high_conf] == y_te[high_conf]).mean()
        print(f"\n  '확신 0.9+' 예측 중 실제 정답: {acc_high:.4f} (확신도와 비슷해야 calibrated)")
        print(f"    {'[OK] Well-calibrated' if abs(acc_high - 0.95) < 0.05 else '[WARN] Miscalibrated' if acc_high < 0.85 else 'OK'}")

    # 5. 다른 seed로 분할 테스트 (성능 변동성)
    print("\n" + "=" * 70)
    print("[5] 다른 seed로 재분할 → 결과 변동성")
    print("=" * 70)
    seed_aurocs = []
    for s in [42, 0, 7, 100, 2026]:
        tr_m2, va_m2, te_m2 = patient_level_split(sid, seed=s)
        # 5초 split mask
        tr_2x_s = np.concatenate([tr_m2, tr_m2])
        te_2x_s = np.concatenate([te_m2, te_m2])
        # Train data 양이 비슷한지
        clf2 = LogisticRegression(max_iter=2000, random_state=s, C=1.0,
                                   class_weight="balanced", solver="lbfgs")
        clf2.fit(emb[tr_2x_s], lbl_2x[tr_2x_s])
        prob2 = clf2.predict_proba(emb[te_2x_s])
        y2_bin = label_binarize(lbl_2x[te_2x_s], classes=[0, 1, 2])
        auroc2 = roc_auc_score(y2_bin, prob2, average="macro", multi_class="ovr")
        seed_aurocs.append((s, auroc2))
        print(f"    seed {s:5d} → AUROC macro: {auroc2:.4f} (test n={te_2x_s.sum():,})")
    sa = np.array([a for _, a in seed_aurocs])
    print(f"  seed 변동: mean={sa.mean():.4f}, std={sa.std():.4f}")
    if sa.std() < 0.005:
        print(f"  [OK] 안정적 (std<0.005)")
    elif sa.std() < 0.015:
        print(f"  OK (std<0.015)")
    else:
        print(f"  [WARN] 변동성 큼")

    print("\n[done]")


if __name__ == "__main__":
    main()
