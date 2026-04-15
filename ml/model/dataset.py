"""
HDF5에서 ECG 데이터를 로드하는 PyTorch Dataset
메모리에 전부 올리지 않고 인덱스 기반으로 필요한 만큼만 읽음
"""
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    """
    HDF5 기반 ECG 데이터셋.
    label/numeric/patient는 메모리에 캐싱, waveform만 HDF5에서 읽음.
    """

    def __init__(self, h5_path: str, indices: np.ndarray = None, lead_indices: list = None, use_bp: bool = True):
        """
        Args:
            h5_path: 전처리된 HDF5 파일 경로
            indices: 사용할 인덱스 배열 (train/val/test 분할용)
            lead_indices: 사용할 리드 인덱스 (None이면 전체 12-lead)
            use_bp: BP 피처 사용 여부 (기본: True)
        """
        self.h5_path = h5_path
        self._h5 = None
        self.lead_indices = lead_indices
        self.add_lead3 = False
        self.use_bp = use_bp

        with h5py.File(h5_path, "r") as f:
            total_len = f["label"].shape[0]

        self.indices = indices if indices is not None else np.arange(total_len)

        # 작은 피처들은 메모리에 캐싱 (label ~3MB, numeric ~12MB, patient ~6MB)
        with h5py.File(h5_path, "r") as f:
            all_idx = self.indices
            sorted_order = np.argsort(all_idx)
            sorted_idx = all_idx[sorted_order]

            raw_label = f["label"][sorted_idx]
            raw_numeric = f["numeric_features"][sorted_idx]
            raw_patient = f["patient_features"][sorted_idx]
            has_bp = use_bp and "bp_features" in f
            if has_bp:
                raw_bp = f["bp_features"][sorted_idx]

            # 원래 순서로 복원
            reverse_order = np.argsort(sorted_order)
            self.labels = torch.tensor(raw_label[reverse_order], dtype=torch.long)
            numeric = raw_numeric[reverse_order].astype(np.float32)
            patient = raw_patient[reverse_order].astype(np.float32)
            np.nan_to_num(numeric, copy=False, nan=0.0)
            np.nan_to_num(patient, copy=False, nan=0.0)
            if has_bp:
                bp = raw_bp[reverse_order].astype(np.float32)
                np.nan_to_num(bp, copy=False, nan=0.0)
                # numeric(4) + bp(2) = (6,)
                numeric = np.concatenate([numeric, bp], axis=1)
            self.numeric = torch.from_numpy(numeric)
            self.patient = torch.from_numpy(patient)

    @property
    def h5(self):
        """Lazy open - 멀티프로세스 DataLoader 호환용"""
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", rdcc_nbytes=64 * 1024 * 1024)
        return self._h5

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        waveform = torch.tensor(self.h5["waveform"][real_idx], dtype=torch.float32)
        if self.lead_indices is not None:
            waveform = waveform[:, self.lead_indices]  # (5000, 12) → (5000, N)
        if self.add_lead3:
            # Lead III = Lead II - Lead I (아인트호벤 삼각법)
            lead3 = waveform[:, 1:2] - waveform[:, 0:1]  # (5000, 1)
            waveform = torch.cat([waveform, lead3], dim=1)  # (5000, 3)
        return waveform, self.numeric[idx], self.patient[idx], self.labels[idx]

    def __del__(self):
        if self._h5 is not None:
            self._h5.close()


def split_by_patient(h5_path: str, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    환자 단위로 train/val/test 분할.
    같은 환자의 ECG가 다른 세트에 섞이지 않도록 보장.

    Returns:
        (train_indices, val_indices, test_indices)
    """
    rng = np.random.RandomState(seed)

    with h5py.File(h5_path, "r") as f:
        subject_ids = f["subject_id"][:]

    # 고유 환자 ID 추출
    unique_patients = np.unique(subject_ids)
    rng.shuffle(unique_patients)

    # 환자 단위 분할
    n = len(unique_patients)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_patients = set(unique_patients[:n_train])
    val_patients = set(unique_patients[n_train:n_train + n_val])
    test_patients = set(unique_patients[n_train + n_val:])

    # 인덱스 매핑
    train_idx = np.where(np.isin(subject_ids, list(train_patients)))[0]
    val_idx = np.where(np.isin(subject_ids, list(val_patients)))[0]
    test_idx = np.where(np.isin(subject_ids, list(test_patients)))[0]

    return train_idx, val_idx, test_idx


def compute_class_weights(h5_path: str, indices: np.ndarray, n_classes=3):
    """
    클래스 불균형 처리를 위한 가중치 계산.
    weight = total / (n_classes * count_per_class)
    """
    with h5py.File(h5_path, "r") as f:
        labels = f["label"][:]
    labels = labels[indices]

    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    weights = len(labels) / (n_classes * counts)
    # 0 방지
    weights[counts == 0] = 1.0
    return torch.tensor(weights, dtype=torch.float32)
