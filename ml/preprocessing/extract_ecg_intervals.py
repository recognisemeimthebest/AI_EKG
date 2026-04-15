"""
machine_measurements.csv에서 ECG interval 피처 추출

피처 (5개):
  0. p_duration  - P-wave duration (ms): p_end - p_onset
  1. pr_interval - PR interval (ms): qrs_onset - p_onset
  2. qtc         - QTc (ms): Bazett 공식 QT / sqrt(RR/1000)
  3. p_axis_abnormal - P-axis 이상 플래그 (0~75도 밖이면 1)
  4. qrs_t_angle - QRS-T angle (도): |qrs_axis - t_axis|

기존 ecg_preprocessed.h5에 ecg_interval_features(N, 5) 데이터셋 추가
"""
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

MACHINE_CSV = "g:/AIEKG/data/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/machine_measurements.csv"
H5_PATH = "g:/AIEKG/ml/data/ecg_preprocessed.h5"
N_FEATURES = 5


def main():
    # 1) HDF5에서 study_id 로드
    print("[1/3] HDF5 study_id 로드...")
    with h5py.File(H5_PATH, "r") as f:
        h5_study_ids = f["study_id"][:]
    print(f"  HDF5 레코드: {len(h5_study_ids):,}")

    # 2) machine_measurements 로드
    print("[2/3] machine_measurements.csv 로드...")
    cols = ["study_id", "rr_interval", "p_onset", "p_end",
            "qrs_onset", "qrs_end", "t_end", "p_axis", "qrs_axis", "t_axis"]
    df = pd.read_csv(MACHINE_CSV, usecols=cols)
    print(f"  CSV 레코드: {len(df):,}")

    # study_id 기준 dict 생성
    df = df.drop_duplicates(subset="study_id", keep="first")
    df = df.set_index("study_id")
    print(f"  유니크 study_id: {len(df):,}")

    # 3) 피처 계산
    print("[3/3] 피처 계산 중...")
    features = np.full((len(h5_study_ids), N_FEATURES), np.nan, dtype=np.float32)
    n_matched = 0

    for i in tqdm(range(len(h5_study_ids)), desc="매칭"):
        sid = h5_study_ids[i]
        if sid not in df.index:
            continue

        row = df.loc[sid]
        n_matched += 1

        p_onset = row.get("p_onset", np.nan)
        p_end = row.get("p_end", np.nan)
        qrs_onset = row.get("qrs_onset", np.nan)
        qrs_end = row.get("qrs_end", np.nan)
        t_end = row.get("t_end", np.nan)
        rr = row.get("rr_interval", np.nan)
        p_ax = row.get("p_axis", np.nan)
        qrs_ax = row.get("qrs_axis", np.nan)
        t_ax = row.get("t_axis", np.nan)

        # P-wave duration (정상 범위: 40~200ms)
        if pd.notna(p_onset) and pd.notna(p_end):
            p_dur = p_end - p_onset
            if 20 <= p_dur <= 300:
                features[i, 0] = p_dur

        # PR interval (정상 범위: 80~400ms)
        if pd.notna(p_onset) and pd.notna(qrs_onset):
            pr = qrs_onset - p_onset
            if 50 <= pr <= 500:
                features[i, 1] = pr

        # QTc (Bazett, 정상 범위: 300~600ms)
        if pd.notna(qrs_onset) and pd.notna(t_end) and pd.notna(rr) and rr > 0:
            qt = t_end - qrs_onset
            qtc = qt / np.sqrt(rr / 1000.0)
            if 200 <= qtc <= 700:
                features[i, 2] = qtc

        # P-axis abnormal (0~75도 밖이면 1)
        if pd.notna(p_ax) and -180 <= p_ax <= 360:
            features[i, 3] = 0.0 if (0 <= p_ax <= 75) else 1.0

        # QRS-T angle
        if pd.notna(qrs_ax) and pd.notna(t_ax):
            angle = abs(qrs_ax - t_ax)
            if angle > 180:
                angle = 360 - angle
            features[i, 4] = angle

    print(f"\n매칭: {n_matched:,}/{len(h5_study_ids):,} ({n_matched/len(h5_study_ids)*100:.1f}%)")

    # 통계
    for j, name in enumerate(["p_duration", "pr_interval", "qtc", "p_axis_abnormal", "qrs_t_angle"]):
        valid = ~np.isnan(features[:, j])
        n_valid = valid.sum()
        if n_valid > 0:
            print(f"  {name}: {n_valid:,} valid, mean={np.nanmean(features[:, j]):.1f}")

    # HDF5 저장
    print(f"\nHDF5에 ecg_interval_features 저장...")
    with h5py.File(H5_PATH, "a") as f:
        if "ecg_interval_features" in f:
            del f["ecg_interval_features"]
        f.create_dataset("ecg_interval_features", data=features, dtype="float32")
        f["ecg_interval_features"].attrs["columns"] = [
            "p_duration", "pr_interval", "qtc", "p_axis_abnormal", "qrs_t_angle"
        ]
        f["ecg_interval_features"].attrs["n_matched"] = int(n_matched)
    print("완료!")


if __name__ == "__main__":
    main()
