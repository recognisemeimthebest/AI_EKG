"""
저장된 PAF 실험 모델의 threshold 분석
- val set으로 최적 threshold 탐색 (AF F1 최대화)
- test set에 그 threshold 적용하여 최종 평가
- 기준선 vs focal_loss 비교 출력
"""
import sys
import argparse
import numpy as np
import torch
import h5py
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score,
    precision_recall_fscore_support, classification_report
)
from tqdm import tqdm

# 같은 디렉토리에서 임포트
sys.path.insert(0, str(Path(__file__).parent))
from paf_experiments import PAFDataset, split_by_patient
from resnet34_ecg import ResNet34ECGWithTabular

# ── 설정 ─────────────────────────────────────────────────────
PAF_H5    = "g:/AIEKG/ml/data/ecg_paroxysmal_af.h5"
ECG_H5    = "g:/AIEKG/ml/data/ecg_preprocessed.h5"
CACHE_H5  = "g:/AIEKG/ml/data/ecg_paf_cache.h5"
BATCH     = 64
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINTS = {
    "baseline":   "g:/AIEKG/ml/checkpoints/paroxysmal-af-resnet34-clinical/best_model.pt",
    "focal_loss": "g:/AIEKG/ml/checkpoints/paf-experiments/exp01_focal_loss/best_model.pt",
}


def get_probs(model, loader):
    """모델 추론 → (probs, labels) numpy arrays"""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for waveform, numeric, patient, labels in tqdm(loader, desc="  Inference", leave=False,
                                                        bar_format="{l_bar}{bar:30}{r_bar}"):
            waveform = waveform.to(DEVICE)  # 모델 내부에서 transpose(1,2) 처리
            numeric  = numeric.to(DEVICE)
            patient  = patient.to(DEVICE)
            logits   = model(waveform, numeric, patient)
            probs    = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def find_best_threshold(probs, labels):
    """val set에서 Hidden AF F1 최대 threshold 반환"""
    precisions, recalls, thresholds = precision_recall_curve(labels, probs)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1s[:-1])  # 마지막 원소는 threshold 없음
    return float(thresholds[best_idx]), float(f1s[best_idx])


def evaluate_at_threshold(probs, labels, threshold):
    """threshold 고정 후 P/R/F1 계산"""
    preds = (probs >= threshold).astype(int)
    p_n, r_n, f_n, _ = precision_recall_fscore_support(labels, preds, labels=[0], zero_division=0)
    p_af, r_af, f_af, _ = precision_recall_fscore_support(labels, preds, labels=[1], zero_division=0)
    auroc = roc_auc_score(labels, probs)
    return {
        "auroc": auroc,
        "threshold": threshold,
        "normal_p": p_n[0], "normal_r": r_n[0], "normal_f1": f_n[0],
        "af_p": p_af[0], "af_r": r_af[0], "af_f1": f_af[0],
    }


def load_model(ckpt_path, n_numeric, n_patient):
    model = ResNet34ECGWithTabular(
        n_leads=3, n_numeric=n_numeric, n_patient=n_patient,
        n_classes=2, dropout=0.3
    ).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    return model


def print_result(name, val_res, test_res):
    print(f"\n  [{name}]")
    print(f"    Val  → best_thresh={val_res['threshold']:.3f}  AF F1={val_res['af_f1']:.3f}")
    print(f"    Test → AUROC={test_res['auroc']:.4f}  thresh={test_res['threshold']:.3f}")
    print(f"           Normal   P={test_res['normal_p']:.3f}  R={test_res['normal_r']:.3f}  F1={test_res['normal_f1']:.3f}")
    print(f"           Hidden AF P={test_res['af_p']:.3f}  R={test_res['af_r']:.3f}  F1={test_res['af_f1']:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(CHECKPOINTS.keys()),
                        help="평가할 모델 이름 (baseline, focal_loss)")
    parser.add_argument("--no-preload", action="store_true", help="RAM 프리로드 비활성화")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print("데이터 분할 로딩...")
    _, val_idx, test_idx = split_by_patient(PAF_H5)

    preload = not args.no_preload
    print("Val 데이터셋 로딩...")
    val_ds  = PAFDataset(ECG_H5, PAF_H5, val_idx,  [0, 1], preload=preload)
    print("Test 데이터셋 로딩...")
    test_ds = PAFDataset(ECG_H5, PAF_H5, test_idx, [0, 1], preload=preload)

    val_loader  = DataLoader(val_ds,  batch_size=BATCH, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    n_numeric = val_ds.numeric.shape[1]
    n_patient = val_ds.patient.shape[1]

    print(f"\n{'='*60}")
    print("Threshold 분석 결과 (Val → threshold 선택 → Test 평가)")
    print(f"{'='*60}")

    all_results = {}
    for name in args.models:
        if name not in CHECKPOINTS:
            print(f"  [SKIP] {name}: checkpoint 경로 없음")
            continue
        ckpt = CHECKPOINTS[name]
        if not Path(ckpt).exists():
            print(f"  [SKIP] {name}: 파일 없음 ({ckpt})")
            continue

        print(f"\n  [{name}] 모델 로딩: {ckpt}")
        model = load_model(ckpt, n_numeric, n_patient)

        print(f"  [{name}] Val 추론...")
        val_probs, val_labels = get_probs(model, val_loader)
        best_thresh, best_f1 = find_best_threshold(val_probs, val_labels)
        val_res = evaluate_at_threshold(val_probs, val_labels, best_thresh)

        print(f"  [{name}] Test 추론...")
        test_probs, test_labels = get_probs(model, test_loader)
        test_res = evaluate_at_threshold(test_probs, test_labels, best_thresh)

        all_results[name] = {"val": val_res, "test": test_res}
        print_result(name, val_res, test_res)

    # ── 최종 비교표 ─────────────────────────────────────────
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("최종 비교표 (val threshold → test 평가)")
        print(f"{'='*70}")
        print(f"{'모델':>15} {'AUROC':>7} {'Thresh':>7} {'N.P':>6} {'N.R':>6} {'AF.P':>6} {'AF.R':>6} {'AF.F1':>7}")
        print("-" * 70)
        # 기준선 (threshold=0.5 고정) 참고값
        print(f"{'baseline(0.5)':>15} {'0.824':>7} {'0.500':>7} {'?':>6} {'?':>6} {'0.250':>6} {'0.710':>6} {'0.372':>7}  <- 참고")
        for name, res in all_results.items():
            t = res["test"]
            print(f"{name:>15} {t['auroc']:>7.4f} {t['threshold']:>7.3f} "
                  f"{t['normal_p']:>6.3f} {t['normal_r']:>6.3f} "
                  f"{t['af_p']:>6.3f} {t['af_r']:>6.3f} {t['af_f1']:>7.3f}")


if __name__ == "__main__":
    main()
