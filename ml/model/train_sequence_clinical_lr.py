"""
2단계 접근: 시퀀스 TCN 확률 출력 + 임상정보 + interval 피처 → 로지스틱 회귀
(Khurshid 2022, Jabbour 2024 방식)

1단계: 학습된 시퀀스 TCN으로 각 샘플의 예측 확률 추출
2단계: 확률 + clinical flags + interval features → sklearn LogisticRegression
"""
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from tqdm import tqdm
from pathlib import Path
import h5py

from cnn_tcn import CNNTCN
from train_sequence import SequenceTCN
from sequence_dataset import (
    SequenceDataset, split_by_patient_sequence
)


@torch.no_grad()
def extract_probabilities(model, loader, device):
    """학습된 모델에서 확률과 라벨 추출"""
    model.eval()
    all_probs, all_labels = [], []
    pbar = tqdm(loader, desc="  확률 추출", leave=False)
    for waveforms, numerics, patients, mask, time_gaps, deltas, clinical, labels in pbar:
        waveforms = waveforms.to(device)
        numerics = numerics.to(device)
        patients = patients.to(device)
        mask = mask.to(device)
        time_gaps = time_gaps.to(device)

        # clinical은 모델에 넣지 않음 (n_clinical=0)
        logits = model(waveforms, numerics, patients, mask, time_gaps)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy())

    return np.array(all_probs), np.array(all_labels)


def get_clinical_features(dataset):
    """데이터셋에서 환자별 clinical features 추출"""
    features = []
    for idx in range(len(dataset)):
        sid = int(dataset.subject_ids[idx])
        if dataset._clinical_lookup is not None:
            clin = dataset._clinical_lookup.get(
                sid, np.zeros(dataset.n_clinical, dtype=np.float32)
            )
        else:
            clin = np.zeros(0, dtype=np.float32)
        features.append(clin)
    return np.array(features)


def get_interval_features(dataset, ecg_h5_path):
    """시퀀스의 마지막 ECG의 interval 피처 추출 + 시퀀스 내 변화량(delta)"""
    with h5py.File(ecg_h5_path, "r") as f:
        all_intervals = f["ecg_interval_features"][:]  # (N_ecg, 5)

    np.nan_to_num(all_intervals, copy=False, nan=0.0)
    interval_names = ["p_duration", "pr_interval", "qtc", "p_axis_abnormal", "qrs_t_angle"]
    n_interval = len(interval_names)

    features = []
    for idx in range(len(dataset)):
        seq = dataset.sequences[idx]
        length = int(dataset.lengths[idx])

        # 마지막 유효 ECG의 interval 피처
        last_idx = int(seq[length - 1]) if length > 0 else -1
        if last_idx >= 0:
            last_feat = all_intervals[last_idx]
        else:
            last_feat = np.zeros(n_interval, dtype=np.float32)

        # delta: 마지막 ECG - 첫 ECG (시퀀스 내 변화량)
        first_idx = int(seq[0]) if length > 0 else -1
        if first_idx >= 0 and last_idx >= 0 and length > 1:
            delta_feat = all_intervals[last_idx] - all_intervals[first_idx]
        else:
            delta_feat = np.zeros(n_interval, dtype=np.float32)

        features.append(np.concatenate([last_feat, delta_feat]))

    return np.array(features), interval_names


def main():
    parser = argparse.ArgumentParser(description="2단계: ECG 확률 + 임상정보 → 로지스틱 회귀")
    parser.add_argument("--ecg-data", default="g:/AIEKG/ml/data/ecg_preprocessed.h5")
    parser.add_argument("--seq-data", default="g:/AIEKG/ml/data/ecg_sequence_15d_rhythm.h5")
    parser.add_argument("--backbone", default="g:/AIEKG/ml/checkpoints/cnn-tcn-3lead/best_model.pt")
    parser.add_argument("--seq-model", default="g:/AIEKG/ml/checkpoints/sequence-15d-rhythm-clean/best_model.pt",
                        help="학습된 시퀀스 TCN 체크포인트")
    parser.add_argument("--clinical-data", default="g:/AIEKG/ml/data/clinical_features.h5")
    parser.add_argument("--output-dir", default="g:/AIEKG/ml/checkpoints/sequence-15d-rhythm-clinical-lr")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ====== 1단계: 확률 추출 ======
    print("\n[1/3] 모델 로드 및 확률 추출...")
    lead_indices = [0, 1]
    train_idx, val_idx, test_idx = split_by_patient_sequence(args.seq_data)

    # clinical 포함한 dataset 생성 (clinical은 LR에서만 사용)
    train_ds = SequenceDataset(args.ecg_data, args.seq_data, train_idx,
                               lead_indices=lead_indices,
                               clinical_h5_path=args.clinical_data)
    val_ds = SequenceDataset(args.ecg_data, args.seq_data, val_idx,
                             lead_indices=lead_indices,
                             clinical_h5_path=args.clinical_data)
    test_ds = SequenceDataset(args.ecg_data, args.seq_data, test_idx,
                              lead_indices=lead_indices,
                              clinical_h5_path=args.clinical_data)
    train_ds.add_lead3 = True
    val_ds.add_lead3 = True
    test_ds.add_lead3 = True

    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             pin_memory=True, num_workers=0)

    # 모델 로드 (clinical 없는 베이스라인)
    n_numeric = train_ds.n_numeric
    n_patient = train_ds.n_patient
    backbone = CNNTCN(n_leads=3, n_numeric=n_numeric, n_classes=3, dropout=args.dropout)
    backbone.load_state_dict(torch.load(args.backbone, weights_only=True))

    model = SequenceTCN(
        backbone, feature_dim=64, tcn_channels=32,
        n_numeric=n_numeric, n_patient=n_patient, n_clinical=0,
        dropout=args.dropout, freeze_backbone=True
    ).to(device)

    print(f"  시퀀스 모델: {args.seq_model}")
    model.load_state_dict(torch.load(args.seq_model, weights_only=True))

    # 확률 추출
    print("  Train set 확률 추출...")
    train_probs, train_labels = extract_probabilities(model, train_loader, device)
    print("  Val set 확률 추출...")
    val_probs, val_labels = extract_probabilities(model, val_loader, device)
    print("  Test set 확률 추출...")
    test_probs, test_labels = extract_probabilities(model, test_loader, device)

    # ECG-only 베이스라인 성능
    ecg_auroc = roc_auc_score(test_labels, test_probs)
    print(f"\n  ECG-only Test AUROC: {ecg_auroc:.4f}")

    # Clinical features 추출
    print("\n  Clinical features 추출...")
    train_clinical = get_clinical_features(train_ds)
    val_clinical = get_clinical_features(val_ds)
    test_clinical = get_clinical_features(test_ds)

    n_clinical = train_clinical.shape[1]
    clinical_names = ["dm", "hf", "mi", "aht"]
    print(f"  Clinical features: {n_clinical}개 {clinical_names}")
    print(f"  Train clinical 양성 비율: {[f'{train_clinical[:, i].mean():.1%}' for i in range(n_clinical)]}")

    # Interval features 추출
    print("\n  Interval features 추출...")
    train_interval, interval_names = get_interval_features(train_ds, args.ecg_data)
    val_interval, _ = get_interval_features(val_ds, args.ecg_data)
    test_interval, _ = get_interval_features(test_ds, args.ecg_data)

    # interval delta names
    delta_names = [f"d_{n}" for n in interval_names]
    all_interval_names = interval_names + delta_names
    print(f"  Interval features: {train_interval.shape[1]}개 {all_interval_names}")

    # ====== 2단계: 로지스틱 회귀 ======
    print("\n[2/3] 로지스틱 회귀 학습...")

    # 피처 조합 실험
    experiments = {
        "ecg_only": lambda p, c, iv: p.reshape(-1, 1),
        "ecg+clinical": lambda p, c, iv: np.column_stack([p, c]),
        "ecg+interval": lambda p, c, iv: np.column_stack([p, iv]),
        "ecg+clinical+interval": lambda p, c, iv: np.column_stack([p, c, iv]),
    }

    results = {}
    for name, build_features in experiments.items():
        X_train = build_features(train_probs, train_clinical, train_interval)
        X_val = build_features(val_probs, val_clinical, val_interval)
        X_test = build_features(test_probs, test_clinical, test_interval)

        # StandardScaler (interval 피처의 scale 차이 보정)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # C 값 튜닝 (val set 기준)
        best_c, best_val_auroc = 1.0, 0
        for c in [0.001, 0.01, 0.1, 1.0, 10.0]:
            lr = LogisticRegression(C=c, max_iter=1000, solver="lbfgs")
            lr.fit(X_train, train_labels)
            val_pred = lr.predict_proba(X_val)[:, 1]
            val_auroc = roc_auc_score(val_labels, val_pred)
            if val_auroc > best_val_auroc:
                best_c = c
                best_val_auroc = val_auroc

        # 최종 학습 (best C)
        lr = LogisticRegression(C=best_c, max_iter=1000, solver="lbfgs")
        lr.fit(X_train, train_labels)
        test_pred_prob = lr.predict_proba(X_test)[:, 1]
        test_pred = lr.predict(X_test)
        test_auroc = roc_auc_score(test_labels, test_pred_prob)

        class_names = ["No Event", "Rhythm Abnormal"]
        report = classification_report(test_labels, test_pred, target_names=class_names)
        cm = confusion_matrix(test_labels, test_pred)

        results[name] = {
            "test_auroc": test_auroc,
            "best_c": best_c,
            "val_auroc": best_val_auroc,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }

        print(f"\n  [{name}] C={best_c}, Val AUROC={best_val_auroc:.4f}, Test AUROC={test_auroc:.4f}")
        print(report)
        print(f"  Confusion Matrix:\n{cm}")

        # 계수 출력 (가장 큰 모델)
        if name == "ecg+clinical+interval":
            coef_names = ["ecg_prob"] + clinical_names + all_interval_names
            print(f"\n  계수:")
            print(f"    intercept: {lr.intercept_[0]:.4f}")
            for cn, coef in zip(coef_names, lr.coef_[0]):
                print(f"    {cn}: {coef:+.4f}")

    # ====== 3단계: 결과 저장 ======
    print(f"\n[3/3] 결과 저장: {output_dir}")

    # 비교 테이블
    print("\n" + "=" * 60)
    print("  모델 비교")
    print("=" * 60)
    print(f"  {'모델':<20} {'Test AUROC':>12}")
    print(f"  {'-'*20} {'-'*12}")
    for name, r in results.items():
        print(f"  {name:<20} {r['test_auroc']:>12.4f}")
    print("=" * 60)

    # JSON 저장
    save_results = {
        "ecg_model": args.seq_model,
        "clinical_data": args.clinical_data,
        "experiments": {k: {kk: vv for kk, vv in v.items() if kk != "classification_report"}
                       for k, v in results.items()},
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(save_results, f, indent=2, default=str)

    print("  완료!")


if __name__ == "__main__":
    main()
