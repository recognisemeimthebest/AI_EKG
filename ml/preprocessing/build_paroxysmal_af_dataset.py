"""
발작성 AF 감지 데이터셋 구축

양성: AF 진단 환자의 정상 리듬(Sinus) ECG (진단일 +-90일 이내)
음성: AF 진단 이력 없는 환자의 정상 리듬 ECG

입력: ecg_preprocessed.h5 + MIMIC-IV DB (diagnoses_icd, admissions)
출력: ecg_paroxysmal_af.h5
  - indices: 기존 HDF5의 인덱스
  - paf_label: 0=진짜 정상, 1=숨은 AF
  - subject_id, study_id
"""
import argparse
import numpy as np
import pandas as pd
import h5py
import psycopg2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import DB_CONFIG

INPUT_H5 = "g:/AIEKG/ml/data/ecg_preprocessed.h5"
MACHINE_CSV = "g:/AIEKG/data/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/machine_measurements.csv"
OUTPUT_H5 = "g:/AIEKG/ml/data/ecg_paroxysmal_af.h5"


def get_af_patients_with_dates(cur):
    """AF 진단 환자의 subject_id + 가장 빠른 진단일 추출"""
    cur.execute("""
        SELECT d.subject_id, MIN(a.admittime) AS first_af_date
        FROM mimiciv_hosp.diagnoses_icd d
        JOIN mimiciv_hosp.admissions a ON d.hadm_id = a.hadm_id
        WHERE (d.icd_code LIKE 'I48%%' AND d.icd_version = 10)
           OR (d.icd_code LIKE '42731%%' AND d.icd_version = 9)
        GROUP BY d.subject_id
    """)
    rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["subject_id", "first_af_date"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=90,
                        help="AF 진단일 기준 +-윈도우 (일, 기본 90)")
    args = parser.parse_args()
    WINDOW = args.window

    print(f"[1/5] DB에서 AF 환자 추출 (+-{WINDOW}일 윈도우)...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    af_df = get_af_patients_with_dates(cur)
    af_df["first_af_date"] = pd.to_datetime(af_df["first_af_date"])
    print(f"  AF 환자: {len(af_df):,}명")
    af_subject_ids = set(af_df["subject_id"].values)
    conn.close()

    print("[2/5] HDF5 + machine_measurements 로드...")
    with h5py.File(INPUT_H5, "r") as f:
        h5_subject_ids = f["subject_id"][:]
        h5_study_ids = f["study_id"][:]
        h5_labels = f["label"][:]

    # machine_measurements에서 리듬 정보 가져오기
    mm = pd.read_csv(MACHINE_CSV, usecols=["study_id", "report_0", "ecg_time"])
    mm["ecg_time"] = pd.to_datetime(mm["ecg_time"])

    # HDF5 인덱스 DataFrame
    h5_df = pd.DataFrame({
        "h5_index": np.arange(len(h5_subject_ids)),
        "subject_id": h5_subject_ids,
        "study_id": h5_study_ids,
        "label": h5_labels,
    })

    # machine_measurements와 조인 (ecg_time, report_0 가져오기)
    h5_df = h5_df.merge(mm[["study_id", "report_0", "ecg_time"]], on="study_id", how="left")

    print(f"  전체 ECG: {len(h5_df):,}")
    print(f"  report_0 매칭: {h5_df['report_0'].notna().sum():,}")

    print("[3/5] 정상 리듬(Sinus) ECG 필터링...")
    # Sinus 리듬만 선택
    sinus_mask = h5_df["report_0"].str.lower().str.contains("sinus", na=False)
    sinus_df = h5_df[sinus_mask].copy()
    print(f"  Sinus ECG: {len(sinus_df):,}")

    print("[4/5] 양성/음성 라벨링...")
    # AF 환자 여부
    sinus_df["is_af_patient"] = sinus_df["subject_id"].isin(af_subject_ids)

    # AF 환자: 진단일 +-윈도우 내 ECG만 양성으로
    af_sinus = sinus_df[sinus_df["is_af_patient"]].copy()
    af_sinus = af_sinus.merge(af_df, on="subject_id", how="left")
    af_sinus["days_from_af"] = (af_sinus["ecg_time"] - af_sinus["first_af_date"]).dt.total_seconds() / 86400
    af_sinus["in_window"] = af_sinus["days_from_af"].abs() <= WINDOW

    positive = af_sinus[af_sinus["in_window"]]
    print(f"  양성 (AF 환자 Sinus, +-{WINDOW}일): {len(positive):,} ECGs, "
          f"{positive['subject_id'].nunique():,} patients")

    # 음성: AF 이력 전혀 없는 환자의 Sinus ECG
    negative = sinus_df[~sinus_df["is_af_patient"]]
    print(f"  음성 (Non-AF Sinus): {len(negative):,} ECGs, "
          f"{negative['subject_id'].nunique():,} patients")

    # 합치기
    pos_indices = positive["h5_index"].values
    neg_indices = negative["h5_index"].values

    all_indices = np.concatenate([pos_indices, neg_indices])
    all_labels = np.concatenate([
        np.ones(len(pos_indices), dtype=np.int64),
        np.zeros(len(neg_indices), dtype=np.int64),
    ])
    all_subject_ids = np.concatenate([
        positive["subject_id"].values,
        negative["subject_id"].values,
    ])
    all_study_ids = np.concatenate([
        positive["study_id"].values,
        negative["study_id"].values,
    ])

    # 셔플
    perm = np.random.RandomState(42).permutation(len(all_indices))
    all_indices = all_indices[perm]
    all_labels = all_labels[perm]
    all_subject_ids = all_subject_ids[perm]
    all_study_ids = all_study_ids[perm]

    print(f"\n[5/5] HDF5 저장: {OUTPUT_H5}")
    print(f"  Total: {len(all_indices):,} (pos: {all_labels.sum():,}, "
          f"neg: {(all_labels == 0).sum():,}, ratio: 1:{(all_labels == 0).sum() / max(all_labels.sum(), 1):.1f})")

    with h5py.File(OUTPUT_H5, "w") as f:
        f.create_dataset("indices", data=all_indices)
        f.create_dataset("paf_label", data=all_labels)
        f.create_dataset("subject_id", data=all_subject_ids)
        f.create_dataset("study_id", data=all_study_ids)

    print("Done!")


if __name__ == "__main__":
    main()
