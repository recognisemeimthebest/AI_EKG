"""
시퀀스 예측용 Dataset
각 샘플: 환자의 연속 ECG 시퀀스 → 리듬 이상 예측
"""
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, ecg_h5_path: str, seq_h5_path: str,
                 indices: np.ndarray = None, lead_indices: list = None,
                 clinical_h5_path: str = None, use_interval: bool = False,
                 use_hrv: bool = False):
        self.ecg_h5_path = ecg_h5_path
        self._ecg_h5 = None
        self.lead_indices = lead_indices
        self.add_lead3 = False
        self.use_interval = use_interval
        self.use_hrv = use_hrv

        with h5py.File(seq_h5_path, "r") as f:
            all_sequences = f["sequences"][:]
            all_time_gaps = f["time_gaps"][:] if "time_gaps" in f else np.zeros_like(all_sequences, dtype=np.float32)
            all_lengths = f["seq_lengths"][:]
            all_labels = f["pred_label"][:]
            all_sids = f["subject_id"][:]
            self.max_seq_len = int(f.attrs["max_seq_len"])

        if indices is not None:
            self.sequences = all_sequences[indices]
            self.time_gaps = all_time_gaps[indices]
            self.lengths = all_lengths[indices]
            self.labels = torch.tensor(all_labels[indices], dtype=torch.long)
            self.subject_ids = all_sids[indices]
        else:
            self.sequences = all_sequences
            self.time_gaps = all_time_gaps
            self.lengths = all_lengths
            self.labels = torch.tensor(all_labels, dtype=torch.long)
            self.subject_ids = all_sids

        # numeric/patient/waveform 캐싱 (시퀀스 내 모든 유효 인덱스)
        all_ecg_indices = np.unique(self.sequences[self.sequences >= 0])
        sorted_idx = np.sort(all_ecg_indices)
        print(f"    캐싱: {len(sorted_idx):,}개 ECG 로드 중...")
        with h5py.File(ecg_h5_path, "r") as f:
            self._numeric_cache = {}
            self._patient_cache = {}
            self._waveform_cache = {}
            raw_num = f["numeric_features"][sorted_idx]
            raw_pat = f["patient_features"][sorted_idx]

            # interval features (p_duration, pr_interval, qtc, p_axis_abnormal, qrs_t_angle)
            if use_interval and "ecg_interval_features" in f:
                raw_interval = f["ecg_interval_features"][sorted_idx]
                # NaN → 컬럼별 중앙값으로 대체 (0 대신 median: delta 계산 시 허위 급변 방지)
                # p_duration 71.6% valid, pr_interval 84.7% valid → 0 채우면 delta 왜곡
                for col in range(raw_interval.shape[1]):
                    col_vals = raw_interval[:, col]
                    nan_mask = np.isnan(col_vals)
                    if nan_mask.any():
                        median_val = float(np.nanmedian(col_vals))
                        col_vals[nan_mask] = median_val
                np.nan_to_num(raw_interval, copy=False, nan=0.0)  # 혹시 남은 nan 처리
                raw_num = np.concatenate([raw_num, raw_interval], axis=1)
                print(f"    interval 피처 {raw_interval.shape[1]}개 추가 → numeric {raw_num.shape[1]}개")

            # HRV features (mean_rr, sdnn, rmssd, pnn50, mean_hr, hr_std) + SD1, SD2
            # Gregoire 2025: HRV delta features가 AF 예측에 핵심 (AUROC 0.919)
            # SD1 = rmssd/sqrt(2), SD2 = sqrt(2*sdnn^2 - 0.5*rmssd^2) (Poincare plot)
            if use_hrv and "hrv_features" in f:
                raw_hrv = f["hrv_features"][sorted_idx]  # (N, 6): mean_rr, sdnn, rmssd, pnn50, mean_hr, hr_std
                np.nan_to_num(raw_hrv, copy=False, nan=0.0)
                # SD1/SD2 on-the-fly
                rmssd = raw_hrv[:, 2:3]
                sdnn  = raw_hrv[:, 1:2]
                sd1 = rmssd / np.sqrt(2)
                sd2 = np.sqrt(np.maximum(2.0 * sdnn**2 - 0.5 * rmssd**2, 0.0))
                raw_hrv_ext = np.concatenate([raw_hrv, sd1, sd2], axis=1)  # (N, 8)
                raw_num = np.concatenate([raw_num, raw_hrv_ext], axis=1)
                print(f"    HRV 피처 8개 추가 (6+SD1+SD2) → numeric {raw_num.shape[1]}개")

            # waveform 배치 로드 (HDF5 순차 읽기로 최적화)
            chunk = 10000
            for start in range(0, len(sorted_idx), chunk):
                end = min(start + chunk, len(sorted_idx))
                batch_idx = sorted_idx[start:end]
                raw_wav = f["waveform"][batch_idx]
                for j, idx in enumerate(batch_idx):
                    w = raw_wav[j].astype(np.float32)
                    if lead_indices is not None:
                        w = w[:, lead_indices]
                    self._waveform_cache[int(idx)] = w
            for i, idx in enumerate(sorted_idx):
                n = raw_num[i].astype(np.float32)
                p = raw_pat[i].astype(np.float32)
                np.nan_to_num(n, copy=False, nan=0.0)
                np.nan_to_num(p, copy=False, nan=0.0)
                self._numeric_cache[int(idx)] = n
                self._patient_cache[int(idx)] = p
            self.n_numeric = raw_num.shape[1]
            self.n_patient = raw_pat.shape[1]
        print(f"    캐싱 완료 ({len(self._waveform_cache):,}개)")

        # clinical features 로딩 (환자 단위 플래그)
        if clinical_h5_path is not None:
            with h5py.File(clinical_h5_path, "r") as f:
                clin_sids = f["subject_id"][:]
                clin_flags = f["clinical_flags"][:]
            self._clinical_lookup = {}
            for i, sid in enumerate(clin_sids):
                self._clinical_lookup[int(sid)] = clin_flags[i].astype(np.float32)
            self.n_clinical = clin_flags.shape[1]
        else:
            self._clinical_lookup = None
            self.n_clinical = 0

    def __len__(self):
        return len(self.sequences)

    def _load_single_ecg(self, ecg_idx):
        """단일 ECG 로드: 캐시에서 waveform + numeric + patient"""
        waveform = torch.tensor(self._waveform_cache[ecg_idx], dtype=torch.float32)
        if self.add_lead3:
            lead3 = waveform[:, 1:2] - waveform[:, 0:1]
            waveform = torch.cat([waveform, lead3], dim=1)
        numeric = torch.tensor(self._numeric_cache[ecg_idx], dtype=torch.float32)
        patient = torch.tensor(self._patient_cache[ecg_idx], dtype=torch.float32)
        return waveform, numeric, patient

    def __getitem__(self, idx):
        seq = self.sequences[idx]  # (max_seq_len,)
        length = int(self.lengths[idx])

        # 시퀀스 내 각 ECG 로드
        waveforms = []
        numerics = []
        patients = []
        mask = []

        for i in range(self.max_seq_len):
            ecg_idx = int(seq[i])
            if ecg_idx >= 0:
                w, n, p = self._load_single_ecg(ecg_idx)
                waveforms.append(w)
                numerics.append(n)
                patients.append(p)
                mask.append(1.0)
            else:
                # 패딩: 0으로 채움
                n_leads = 3 if self.add_lead3 else (len(self.lead_indices) if self.lead_indices else 12)
                waveforms.append(torch.zeros(5000, n_leads))
                numerics.append(torch.zeros(self.n_numeric))
                patients.append(torch.zeros(self.n_patient))
                mask.append(0.0)

        waveforms = torch.stack(waveforms)    # (seq_len, 5000, n_leads)
        numerics = torch.stack(numerics)      # (seq_len, n_numeric)
        patients = torch.stack(patients)      # (seq_len, n_patient)
        mask = torch.tensor(mask, dtype=torch.float32)  # (seq_len,)
        time_gaps = torch.tensor(self.time_gaps[idx], dtype=torch.float32)  # (seq_len,)

        # delta features: 연속 ECG 간 numeric 변화량 (Suzuki 방식)
        # deltas[0] = 0 (이전 ECG 없음), deltas[t] = numerics[t] - numerics[t-1]
        deltas = torch.zeros_like(numerics)  # (seq_len, n_numeric)
        for t in range(1, self.max_seq_len):
            if mask[t] > 0 and mask[t - 1] > 0:
                deltas[t] = numerics[t] - numerics[t - 1]

        # clinical features (환자 단위, 시퀀스 내 동일)
        if self._clinical_lookup is not None:
            sid = int(self.subject_ids[idx])
            clinical = torch.tensor(
                self._clinical_lookup.get(sid, np.zeros(self.n_clinical, dtype=np.float32)),
                dtype=torch.float32
            )
        else:
            clinical = torch.zeros(0, dtype=torch.float32)

        return waveforms, numerics, patients, mask, time_gaps, deltas, clinical, self.labels[idx]

    def __del__(self):
        pass


def split_by_patient_sequence(seq_h5_path: str, train_ratio=0.7, val_ratio=0.15, seed=42):
    rng = np.random.RandomState(seed)
    with h5py.File(seq_h5_path, "r") as f:
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
    test_idx = np.where(np.isin(subject_ids, list(set(unique_patients[n_train + n_val:]))))[0]
    return train_idx, val_idx, test_idx


def compute_class_weights_sequence(seq_h5_path: str, indices: np.ndarray, n_classes=2):
    with h5py.File(seq_h5_path, "r") as f:
        labels = f["pred_label"][:]
    labels = labels[indices]
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    weights = len(labels) / (n_classes * counts)
    weights[counts == 0] = 1.0
    return torch.tensor(weights, dtype=torch.float32)
