"""
ECG 파형에서 HRV (심박변이도) 피처 추출

R-peak 검출 → RR 간격 → HRV 지표 계산
기존 ecg_preprocessed.h5에 hrv_features(N, 6) 데이터셋 추가

피처:
  0. mean_rr   - 평균 RR 간격 (ms)
  1. sdnn      - RR 간격 표준편차 (ms)
  2. rmssd     - 연속 RR 차이의 RMS (ms)
  3. pnn50     - 연속 RR 차이 >50ms 비율
  4. mean_hr   - 평균 심박수 (bpm)
  5. hr_std    - 심박수 표준편차
"""
import numpy as np
import h5py
from scipy.signal import find_peaks
from tqdm import tqdm
import argparse

SAMPLING_RATE = 500  # Hz (MIMIC-IV ECG)
N_HRV_FEATURES = 6


def detect_r_peaks(ecg_lead, fs=SAMPLING_RATE):
    """Lead II에서 R-peak 검출 (간단한 방법: scipy find_peaks)"""
    # 최소 R-R 간격: 300ms (200bpm), 최소 높이: 신호 중앙값 + 0.5*std
    min_distance = int(0.3 * fs)
    threshold = np.median(ecg_lead) + 0.5 * np.std(ecg_lead)

    peaks, _ = find_peaks(ecg_lead, distance=min_distance, height=threshold)
    return peaks


def compute_hrv(ecg_lead, fs=SAMPLING_RATE):
    """단일 ECG lead에서 HRV 피처 6개 계산"""
    peaks = detect_r_peaks(ecg_lead, fs)

    # R-peak 3개 미만이면 HRV 계산 불가
    if len(peaks) < 3:
        return np.full(N_HRV_FEATURES, np.nan, dtype=np.float32)

    # RR 간격 (ms)
    rr_intervals = np.diff(peaks) / fs * 1000.0

    # 비정상 RR 제거 (300ms~2000ms, 즉 30~200bpm)
    valid = (rr_intervals >= 300) & (rr_intervals <= 2000)
    rr = rr_intervals[valid]

    if len(rr) < 2:
        return np.full(N_HRV_FEATURES, np.nan, dtype=np.float32)

    mean_rr = np.mean(rr)
    sdnn = np.std(rr, ddof=1)
    rr_diff = np.diff(rr)
    rmssd = np.sqrt(np.mean(rr_diff ** 2))
    pnn50 = np.sum(np.abs(rr_diff) > 50) / len(rr_diff)
    hr = 60000.0 / rr
    mean_hr = np.mean(hr)
    hr_std = np.std(hr, ddof=1)

    return np.array([mean_rr, sdnn, rmssd, pnn50, mean_hr, hr_std], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="ECG 파형에서 HRV 피처 추출")
    parser.add_argument("--h5", default="g:/AIEKG/ml/data/ecg_preprocessed.h5")
    parser.add_argument("--lead", type=int, default=1, help="R-peak 검출용 lead index (1=Lead II)")
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()

    with h5py.File(args.h5, "r") as f:
        n_total = f["waveform"].shape[0]
    print(f"총 레코드: {n_total:,}")
    print(f"R-peak 검출 lead: {args.lead} (Lead II)")

    hrv_all = np.full((n_total, N_HRV_FEATURES), np.nan, dtype=np.float32)
    n_valid = 0

    with h5py.File(args.h5, "r", rdcc_nbytes=128 * 1024 * 1024) as f:
        waveforms = f["waveform"]
        for start in tqdm(range(0, n_total, args.batch_size), desc="HRV 추출"):
            end = min(start + args.batch_size, n_total)
            batch = waveforms[start:end]  # (batch, 5000, 12)

            for i in range(batch.shape[0]):
                ecg_lead = batch[i, :, args.lead]
                hrv = compute_hrv(ecg_lead)
                hrv_all[start + i] = hrv
                if not np.isnan(hrv[0]):
                    n_valid += 1

    print(f"\nHRV 계산 완료: {n_valid:,}/{n_total:,} ({n_valid/n_total*100:.1f}%)")
    valid_mask = ~np.isnan(hrv_all[:, 0])
    if n_valid > 0:
        print(f"  Mean RR: {np.nanmean(hrv_all[:, 0]):.1f}ms")
        print(f"  SDNN:    {np.nanmean(hrv_all[:, 1]):.1f}ms")
        print(f"  RMSSD:   {np.nanmean(hrv_all[:, 2]):.1f}ms")
        print(f"  pNN50:   {np.nanmean(hrv_all[:, 3]):.3f}")
        print(f"  Mean HR: {np.nanmean(hrv_all[:, 4]):.1f}bpm")
        print(f"  HR Std:  {np.nanmean(hrv_all[:, 5]):.1f}")

    # HDF5에 저장
    print(f"\nHDF5에 hrv_features 저장 중...")
    with h5py.File(args.h5, "a") as f:
        if "hrv_features" in f:
            del f["hrv_features"]
        f.create_dataset("hrv_features", data=hrv_all, dtype="float32")
        f["hrv_features"].attrs["columns"] = ["mean_rr", "sdnn", "rmssd", "pnn50", "mean_hr", "hr_std"]
        f["hrv_features"].attrs["n_valid"] = int(n_valid)
    print("완료!")


if __name__ == "__main__":
    main()
