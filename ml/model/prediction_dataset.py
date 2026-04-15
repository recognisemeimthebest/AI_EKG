"""
30일 AFib 예측용 Dataset
기존 ecg_preprocessed.h5의 waveform/numeric/patient를 재사용하고
ecg_prediction_30d.h5의 indices/pred_label을 참조
"""
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset


class PredictionDataset(Dataset):
    """30일 AFib 예측 Dataset"""

    def __init__(self, ecg_h5_path: str, pred_h5_path: str,
                 indices: np.ndarray = None, lead_indices: list = None,
                 use_hrv: bool = False, use_intervals: bool = False):
        self.ecg_h5_path = ecg_h5_path
        self._ecg_h5 = None
        self.lead_indices = lead_indices
        self.add_lead3 = False

        # 예측 데이터셋 로드
        with h5py.File(pred_h5_path, "r") as f:
            all_ecg_indices = f["indices"][:]
            all_labels = f["pred_label"][:]
            all_sids = f["subject_id"][:]

        # subset indices
        if indices is not None:
            self.ecg_indices = all_ecg_indices[indices]
            self.labels = torch.tensor(all_labels[indices], dtype=torch.long)
            self.subject_ids = all_sids[indices]
        else:
            self.ecg_indices = all_ecg_indices
            self.labels = torch.tensor(all_labels, dtype=torch.long)
            self.subject_ids = all_sids

        # numeric/patient 캐싱
        with h5py.File(ecg_h5_path, "r") as f:
            sorted_order = np.argsort(self.ecg_indices)
            sorted_idx = self.ecg_indices[sorted_order]

            raw_numeric = f["numeric_features"][sorted_idx]
            raw_patient = f["patient_features"][sorted_idx]

            reverse_order = np.argsort(sorted_order)
            numeric = raw_numeric[reverse_order].astype(np.float32)
            patient = raw_patient[reverse_order].astype(np.float32)
            np.nan_to_num(numeric, copy=False, nan=0.0)
            np.nan_to_num(patient, copy=False, nan=0.0)

            # HRV 피처 결합
            if use_hrv and "hrv_features" in f:
                raw_hrv = f["hrv_features"][sorted_idx]
                hrv = raw_hrv[reverse_order].astype(np.float32)
                np.nan_to_num(hrv, copy=False, nan=0.0)
                numeric = np.concatenate([numeric, hrv], axis=1)

            # ECG interval 피처 결합
            if use_intervals and "ecg_interval_features" in f:
                raw_iv = f["ecg_interval_features"][sorted_idx]
                iv = raw_iv[reverse_order].astype(np.float32)
                np.nan_to_num(iv, copy=False, nan=0.0)
                numeric = np.concatenate([numeric, iv], axis=1)

            self.numeric = torch.from_numpy(numeric)
            self.patient = torch.from_numpy(patient)

    @property
    def ecg_h5(self):
        if self._ecg_h5 is None:
            self._ecg_h5 = h5py.File(self.ecg_h5_path, "r", rdcc_nbytes=64 * 1024 * 1024)
        return self._ecg_h5

    def __len__(self):
        return len(self.ecg_indices)

    def __getitem__(self, idx):
        real_idx = int(self.ecg_indices[idx])
        waveform = torch.tensor(self.ecg_h5["waveform"][real_idx], dtype=torch.float32)
        if self.lead_indices is not None:
            waveform = waveform[:, self.lead_indices]
        if self.add_lead3:
            lead3 = waveform[:, 1:2] - waveform[:, 0:1]
            waveform = torch.cat([waveform, lead3], dim=1)
        return waveform, self.numeric[idx], self.patient[idx], self.labels[idx]

    def __del__(self):
        if self._ecg_h5 is not None:
            self._ecg_h5.close()


def split_by_patient_prediction(pred_h5_path: str, train_ratio=0.7, val_ratio=0.15, seed=42):
    """환자 단위 분할"""
    rng = np.random.RandomState(seed)

    with h5py.File(pred_h5_path, "r") as f:
        subject_ids = f["subject_id"][:]

    unique_patients = np.unique(subject_ids)
    rng.shuffle(unique_patients)

    n = len(unique_patients)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_patients = set(unique_patients[:n_train])
    val_patients = set(unique_patients[n_train:n_train + n_val])

    train_idx = np.where(np.isin(subject_ids, list(train_patients)))[0]
    val_idx = np.where(np.isin(subject_ids, list(val_patients)))[0]
    test_idx = np.where(np.isin(subject_ids, list(val_patients | set(unique_patients[n_train + n_val:]))))[0]
    test_idx = np.where(np.isin(subject_ids, list(set(unique_patients[n_train + n_val:]))))[0]

    return train_idx, val_idx, test_idx


def compute_class_weights_prediction(pred_h5_path: str, indices: np.ndarray, n_classes=2):
    """클래스 가중치 계산"""
    with h5py.File(pred_h5_path, "r") as f:
        labels = f["pred_label"][:]
    labels = labels[indices]

    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    weights = len(labels) / (n_classes * counts)
    weights[counts == 0] = 1.0
    return torch.tensor(weights, dtype=torch.float32)
