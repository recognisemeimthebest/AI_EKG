"""
심부전(Heart Failure) 감지 데이터셋 구축

ICD-10: I50.* (Heart failure)
ICD-9: 428.* (Heart failure)

양성: 입원 중 HF 진단이 있는 환자의 ECG
음성: HF 진단이 없는 환자의 ECG (입원 기간 내 촬영)

입력: ecg_preprocessed.h5 + MIMIC-IV DB (diagnoses_icd, admissions, record_list)
출력: ecg_hf.h5
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
OUTPUT_H5 = "g:/AIEKG/ml/data/ecg_hf.h5"


def main():
    print("[1/4] DB에서 HF 진단 환자 조회...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # HF 진단이 있는 hadm_id 집합
    cur.execute("""
        SELECT DISTINCT hadm_id
        FROM mimiciv_hosp.diagnoses_icd
        WHERE (icd_version = 10 AND icd_code LIKE 'I50%%')
           OR (icd_version = 9 AND icd_code LIKE '428%%')
    """)
    hf_hadm_ids = set(row[0] for row in cur.fetchall())
    print(f"  HF 진단 입원: {len(hf_hadm_ids):,}")

    # ECG-입원 매칭 (입원 기간 내 ECG만)
    print("\n[2/4] ECG-입원 매칭...")
    cur.execute("""
        SELECT r.study_id, r.subject_id, a.hadm_id
        FROM mimiciv_ecg.record_list r
        JOIN mimiciv_hosp.admissions a
            ON r.subject_id = a.subject_id
            AND r.ecg_time BETWEEN a.admittime AND a.dischtime
        ORDER BY r.study_id
    """)
    rows = cur.fetchall()
    conn.close()
    print(f"  ECG-입원 매칭: {len(rows):,}")

    df = pd.DataFrame(rows, columns=["study_id", "subject_id", "hadm_id"])

    # 한 ECG가 여러 입원에 매칭될 수 있으므로 중복 제거
    # HF 양성 입원이 있으면 양성으로 판정
    df["hf_label"] = df["hadm_id"].isin(hf_hadm_ids).astype(int)
    df_pos = df[df["hf_label"] == 1].drop_duplicates(subset="study_id", keep="first")
    df_neg = df[df["hf_label"] == 0].drop_duplicates(subset="study_id", keep="first")
    # 양성에 이미 있는 study_id는 음성에서 제거
    df_neg = df_neg[~df_neg["study_id"].isin(df_pos["study_id"])]
    df = pd.concat([df_pos, df_neg], ignore_index=True)
    print(f"  중복 제거 후: {len(df):,}")

    pos = df[df["hf_label"] == 1]
    neg = df[df["hf_label"] == 0]
    print(f"  HF 양성: {len(pos):,} ECGs, {pos['subject_id'].nunique():,} patients")
    print(f"  HF 음성: {len(neg):,} ECGs, {neg['subject_id'].nunique():,} patients")

    # HDF5 인덱스 매칭
    print("\n[3/4] HDF5 인덱스 매칭...")
    with h5py.File(INPUT_H5, "r") as f:
        h5_study_ids = f["study_id"][:]

    h5_sid_set = set(h5_study_ids.tolist())
    h5_sid_to_idx = {sid: idx for idx, sid in enumerate(h5_study_ids)}

    df = df[df["study_id"].isin(h5_sid_set)]
    df["h5_index"] = df["study_id"].map(h5_sid_to_idx)
    print(f"  HDF5 매칭 후: {len(df):,}")

    pos = df[df["hf_label"] == 1]
    neg = df[df["hf_label"] == 0]
    print(f"  HF 양성: {len(pos):,}, HF 음성: {len(neg):,}")

    # 저장
    print("\n[4/4] HDF5 저장...")
    all_indices = df["h5_index"].values.astype(np.int64)
    all_labels = df["hf_label"].values.astype(np.int64)
    all_subject_ids = df["subject_id"].values.astype(np.int64)
    all_study_ids = df["study_id"].values.astype(np.int64)

    perm = np.random.RandomState(42).permutation(len(all_indices))
    all_indices = all_indices[perm]
    all_labels = all_labels[perm]
    all_subject_ids = all_subject_ids[perm]
    all_study_ids = all_study_ids[perm]

    with h5py.File(OUTPUT_H5, "w") as f:
        f.create_dataset("indices", data=all_indices)
        f.create_dataset("hf_label", data=all_labels)
        f.create_dataset("subject_id", data=all_subject_ids)
        f.create_dataset("study_id", data=all_study_ids)

    n_pos = all_labels.sum()
    n_neg = len(all_labels) - n_pos
    print(f"\n  Total: {len(all_labels):,} (HF: {n_pos:,}, non-HF: {n_neg:,}, ratio 1:{n_neg/max(n_pos,1):.1f})")
    print("Done!")


if __name__ == "__main__":
    main()
