"""
임계값 최적화: 기존 모델의 확률 출력에서 최적 threshold를 찾아 Precision 개선

검증셋에서 최적 임계값 탐색 → 테스트셋에서 최종 평가
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score
)
import argparse
from pathlib import Path

from cnn_tcn import CNNTCN
from train_prediction import AFibPredictor, evaluate
from prediction_dataset import (
    PredictionDataset, split_by_patient_prediction
)


def find_optimal_thresholds(probs, labels):
    """다양한 기준으로 최적 임계값 탐색"""
    precision, recall, thresholds = precision_recall_curve(labels, probs)

    results = {}

    # 1. F1 최대화
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_f1_idx = np.argmax(f1_scores)
    results["max_f1"] = {
        "threshold": float(thresholds[best_f1_idx]),
        "precision": float(precision[best_f1_idx]),
        "recall": float(recall[best_f1_idx]),
        "f1": float(f1_scores[best_f1_idx]),
    }

    # 2. Precision >= 40% 에서 최대 Recall
    mask_p40 = precision >= 0.40
    if mask_p40.any():
        valid_idx = np.where(mask_p40)[0]
        best_recall_idx = valid_idx[np.argmax(recall[valid_idx])]
        results["precision_40"] = {
            "threshold": float(thresholds[min(best_recall_idx, len(thresholds)-1)]),
            "precision": float(precision[best_recall_idx]),
            "recall": float(recall[best_recall_idx]),
        }

    # 3. Precision >= 50% 에서 최대 Recall
    mask_p50 = precision >= 0.50
    if mask_p50.any():
        valid_idx = np.where(mask_p50)[0]
        best_recall_idx = valid_idx[np.argmax(recall[valid_idx])]
        results["precision_50"] = {
            "threshold": float(thresholds[min(best_recall_idx, len(thresholds)-1)]),
            "precision": float(precision[best_recall_idx]),
            "recall": float(recall[best_recall_idx]),
        }

    # 4. 고정 임계값들
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        preds = (probs >= thr).astype(int)
        if preds.sum() > 0:
            p = precision_score(labels, preds, zero_division=0)
            r = recall_score(labels, preds, zero_division=0)
            f1 = f1_score(labels, preds, zero_division=0)
            results[f"thr_{thr:.1f}"] = {
                "threshold": thr,
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1),
                "n_positive_pred": int(preds.sum()),
            }

    return results


def main():
    parser = argparse.ArgumentParser(description="Prediction 임계값 최적화")
    parser.add_argument("--ecg-data", default="g:/AIEKG/ml/data/ecg_preprocessed.h5")
    parser.add_argument("--pred-data", default="g:/AIEKG/ml/data/ecg_prediction_15d.h5")
    parser.add_argument("--model-dir", default="g:/AIEKG/ml/checkpoints/prediction-15d")
    parser.add_argument("--backbone-path", default="g:/AIEKG/ml/checkpoints/cnn-tcn-3lead/best_model.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 데이터
    print("\n[1/3] 데이터 로드...")
    lead_indices = [0, 1]
    train_idx, val_idx, test_idx = split_by_patient_prediction(args.pred_data)

    val_ds = PredictionDataset(args.ecg_data, args.pred_data, val_idx, lead_indices=lead_indices)
    test_ds = PredictionDataset(args.ecg_data, args.pred_data, test_idx, lead_indices=lead_indices)
    val_ds.add_lead3 = True
    test_ds.add_lead3 = True

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # 모델 로드
    print("[2/3] 모델 로드...")
    n_numeric = val_ds.numeric.shape[1]
    backbone = CNNTCN(n_leads=3, n_numeric=n_numeric, n_classes=3, dropout=0.3)
    state = torch.load(args.backbone_path, weights_only=True)
    backbone.load_state_dict(state)

    model = AFibPredictor(backbone, n_numeric=n_numeric, dropout=0.3, freeze_backbone=False).to(device)
    model_state = torch.load(str(Path(args.model_dir) / "best_model.pt"), weights_only=True)
    model.load_state_dict(model_state)

    criterion = nn.CrossEntropyLoss()

    # 검증셋에서 확률 추출
    print("[3/3] 임계값 최적화...")
    _, _, _, val_probs, val_labels = evaluate(model, val_loader, criterion, device, desc="Val")
    _, _, _, test_probs, test_labels = evaluate(model, test_loader, criterion, device, desc="Test")

    val_auroc = roc_auc_score(val_labels, val_probs)
    test_auroc = roc_auc_score(test_labels, test_probs)
    print(f"\n  Val AUROC: {val_auroc:.4f}")
    print(f"  Test AUROC: {test_auroc:.4f}")

    # 검증셋 최적 임계값 탐색
    print("\n" + "=" * 70)
    print("검증셋 임계값 탐색 결과")
    print("=" * 70)
    val_results = find_optimal_thresholds(val_probs, val_labels)
    for name, r in sorted(val_results.items()):
        line = f"  {name:>15}: thr={r['threshold']:.3f}  P={r['precision']:.1%}  R={r['recall']:.1%}"
        if "f1" in r:
            line += f"  F1={r['f1']:.3f}"
        if "n_positive_pred" in r:
            line += f"  (n={r['n_positive_pred']})"
        print(line)

    # 테스트셋에서 주요 임계값 적용
    print("\n" + "=" * 70)
    print("테스트셋 결과 (각 임계값)")
    print("=" * 70)

    class_names = ["No AFib", "AFib <15d"]
    for thr_name in ["thr_0.5", "max_f1", "precision_40", "precision_50"]:
        if thr_name not in val_results:
            continue
        thr = val_results[thr_name]["threshold"]
        preds = (test_probs >= thr).astype(int)

        print(f"\n--- {thr_name} (threshold={thr:.3f}) ---")
        print(classification_report(test_labels, preds, target_names=class_names))
        cm = confusion_matrix(test_labels, preds)
        print(f"  Confusion Matrix:")
        print(f"    TN={cm[0][0]:,}  FP={cm[0][1]:,}")
        print(f"    FN={cm[1][0]:,}  TP={cm[1][1]:,}")
        print(f"  오경보율: {cm[0][1]/(cm[0][1]+cm[1][1])*100:.1f}% (FP / 전체 경고)")


if __name__ == "__main__":
    main()
