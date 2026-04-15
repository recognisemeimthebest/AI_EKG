"""
입원 중 사망 예측 데이터셋 구축

ECG가 입원 기간 내에 촬영된 경우만 사용
양성: hospital_expire_flag = 1 (입원 중 사망)
음성: hospital_expire_flag = 0 (생존 퇴원)

입력: ecg_preprocessed.h5 + MIMIC-IV DB (admissions, record_list)
출력: ecg_mortality.h5
"""
import numpy as np
import pandas as pd
import h5py
import psycopg2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import DB_CONFIG

INPUT_H5 = "g:/AIEKG/ml/data/ecg_preprocessed.h5"
OUTPUT_H5 = "g:/AIEKG/ml/data/ecg_mortality.h5"


def main():
    print("[1/3] DB에서 ECG-입원 매칭...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # ECG가 입원 기간 내인 경우만 매칭
    cur.execute("""
        SELECT r.study_id, r.subject_id, a.hospital_expire_flag, a.hadm_id,
               EXTRACT(EPOCH FROM (a.dischtime - r.ecg_time)) AS time_to_discharge_sec
        FROM mimiciv_ecg.record_list r
        JOIN mimiciv_hosp.admissions a
            ON r.subject_id = a.subject_id
            AND r.ecg_time BETWEEN a.admittime AND a.dischtime
        ORDER BY r.study_id
    """)
    rows = cur.fetchall()
    conn.close()
    print(f"  ECG-입원 매칭: {len(rows):,}")

    df = pd.DataFrame(rows, columns=["study_id", "subject_id", "expire_flag", "hadm_id",
                                      "time_to_discharge"])

    # 한 ECG가 여러 입원에 매칭될 수 있으므로 가장 가까운 입원만 사용
    df["abs_time"] = df["time_to_discharge"].abs()
    df = df.sort_values("abs_time").drop_duplicates(subset="study_id", keep="first")
    df = df.drop(columns=["abs_time"])
    print(f"  중복 제거 후: {len(df):,}")

    died = df[df["expire_flag"] == 1]
    survived = df[df["expire_flag"] == 0]
    print(f"  사망: {len(died):,} ECGs, {died['subject_id'].nunique():,} patients")
    print(f"  생존: {len(survived):,} ECGs, {survived['subject_id'].nunique():,} patients")

    # HDF5 인덱스 매칭
    print("\n[2/3] HDF5 인덱스 매칭...")
    with h5py.File(INPUT_H5, "r") as f:
        h5_study_ids = f["study_id"][:]

    h5_sid_set = set(h5_study_ids.tolist())
    h5_sid_to_idx = {sid: idx for idx, sid in enumerate(h5_study_ids)}

    df = df[df["study_id"].isin(h5_sid_set)]
    df["h5_index"] = df["study_id"].map(h5_sid_to_idx)
    print(f"  HDF5 매칭 후: {len(df):,}")

    died = df[df["expire_flag"] == 1]
    survived = df[df["expire_flag"] == 0]
    print(f"  사망: {len(died):,}, 생존: {len(survived):,}")

    # 데이터셋 조립 + 셔플
    print("\n[3/3] HDF5 저장...")
    all_indices = df["h5_index"].values.astype(np.int64)
    all_labels = df["expire_flag"].values.astype(np.int64)
    all_subject_ids = df["subject_id"].values.astype(np.int64)
    all_study_ids = df["study_id"].values.astype(np.int64)
    all_time_to_discharge = df["time_to_discharge"].values.astype(np.float32)

    perm = np.random.RandomState(42).permutation(len(all_indices))
    all_indices = all_indices[perm]
    all_labels = all_labels[perm]
    all_subject_ids = all_subject_ids[perm]
    all_study_ids = all_study_ids[perm]
    all_time_to_discharge = all_time_to_discharge[perm]

    with h5py.File(OUTPUT_H5, "w") as f:
        f.create_dataset("indices", data=all_indices)
        f.create_dataset("mortality_label", data=all_labels)
        f.create_dataset("subject_id", data=all_subject_ids)
        f.create_dataset("study_id", data=all_study_ids)
        f.create_dataset("time_to_discharge_sec", data=all_time_to_discharge)

    pos = all_labels.sum()
    neg = len(all_labels) - pos
    print(f"\n  Total: {len(all_labels):,} (died: {pos:,}, survived: {neg:,}, ratio 1:{neg/pos:.1f})")
    print("Done!")


if __name__ == "__main__":
    main()
