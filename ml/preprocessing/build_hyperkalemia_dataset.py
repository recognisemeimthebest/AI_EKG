"""
고칼륨혈증 감지 데이터셋 구축

ECG 촬영 +-2시간 내 칼륨 검사 결과를 매칭
양성: K >= 5.5 mEq/L (고칼륨혈증)
음성: K 3.5~5.0 mEq/L (정상)
제외: K < 3.5 (저칼륨) 또는 5.0~5.5 (경계)

입력: ecg_preprocessed.h5 + MIMIC-IV DB (labevents, record_list)
출력: ecg_hyperkalemia.h5
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
OUTPUT_H5 = "g:/AIEKG/ml/data/ecg_hyperkalemia_v2.h5"

# Potassium itemids (Blood)
K_ITEMIDS = [50971, 52610, 50822, 52452]  # Blood Chemistry + Whole Blood


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-hours", type=int, default=2,
                        help="ECG-lab matching window (hours, default 2)")
    parser.add_argument("--train-window-hours", type=int, default=None,
                        help="Training set matching window (hours). If set, uses asymmetric window (Kwon 2024)")
    parser.add_argument("--k-high", type=float, default=5.5,
                        help="Hyperkalemia threshold (default 5.5)")
    parser.add_argument("--k-normal-low", type=float, default=3.5)
    parser.add_argument("--k-normal-high", type=float, default=5.0)
    args = parser.parse_args()

    max_window = args.train_window_hours if args.train_window_hours else args.window_hours
    print(f"[1/4] DB에서 ECG-칼륨 매칭 (+-{max_window}시간)...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # ECG와 칼륨 검사 매칭 (+-window 시간)
    k_ids = ",".join(str(i) for i in K_ITEMIDS)
    cur.execute(f"""
        SELECT r.study_id, r.subject_id, l.valuenum,
               ABS(EXTRACT(EPOCH FROM (l.charttime - r.ecg_time))) AS time_diff_sec
        FROM mimiciv_ecg.record_list r
        JOIN mimiciv_hosp.labevents l ON r.subject_id = l.subject_id
        WHERE l.itemid IN ({k_ids})
        AND l.valuenum IS NOT NULL
        AND l.valuenum BETWEEN 1.0 AND 12.0
        AND ABS(EXTRACT(EPOCH FROM (l.charttime - r.ecg_time))) <= {max_window * 3600}
        ORDER BY r.study_id, time_diff_sec
    """)
    rows = cur.fetchall()
    conn.close()
    print(f"  총 매칭: {len(rows):,}")

    # study_id별 가장 가까운 칼륨값 선택
    matched = {}
    for study_id, subject_id, k_value, time_diff in rows:
        if study_id not in matched or time_diff < matched[study_id][2]:
            matched[study_id] = (subject_id, k_value, time_diff)

    print(f"  고유 ECG-칼륨 쌍: {len(matched):,}")

    # 라벨링
    df = pd.DataFrame([
        {"study_id": sid, "subject_id": v[0], "k_value": v[1], "time_diff": v[2]}
        for sid, v in matched.items()
    ])

    normal_mask = (df["k_value"] >= args.k_normal_low) & (df["k_value"] <= args.k_normal_high)
    hyper_mask = df["k_value"] >= args.k_high

    normal_df = df[normal_mask].copy()
    hyper_df = df[hyper_mask].copy()

    print(f"\n  정상 (K {args.k_normal_low}-{args.k_normal_high}): {len(normal_df):,} ECGs, "
          f"{normal_df['subject_id'].nunique():,} patients")
    print(f"  고칼륨 (K >= {args.k_high}): {len(hyper_df):,} ECGs, "
          f"{hyper_df['subject_id'].nunique():,} patients")

    # 고칼륨 세부 분류
    mild = df[(df["k_value"] >= 5.5) & (df["k_value"] < 6.0)]
    moderate = df[(df["k_value"] >= 6.0) & (df["k_value"] < 6.5)]
    severe = df[df["k_value"] >= 6.5]
    print(f"    중등도 (5.5-6.0): {len(mild):,}")
    print(f"    중증 (6.0-6.5): {len(moderate):,}")
    print(f"    위험 (>6.5): {len(severe):,}")

    print(f"\n[2/4] HDF5 인덱스 매핑...")
    with h5py.File(INPUT_H5, "r") as f:
        h5_study_ids = f["study_id"][:]

    h5_study_set = {int(sid): idx for idx, sid in enumerate(h5_study_ids)}

    # HDF5에 있는 ECG만 필터
    normal_df["h5_index"] = normal_df["study_id"].map(h5_study_set)
    hyper_df["h5_index"] = hyper_df["study_id"].map(h5_study_set)
    normal_df = normal_df.dropna(subset=["h5_index"])
    hyper_df = hyper_df.dropna(subset=["h5_index"])
    normal_df["h5_index"] = normal_df["h5_index"].astype(int)
    hyper_df["h5_index"] = hyper_df["h5_index"].astype(int)

    print(f"  HDF5 매칭 - 정상: {len(normal_df):,}, 고칼륨: {len(hyper_df):,}")

    print(f"\n[3/4] 데이터셋 구성...")
    all_indices = np.concatenate([
        normal_df["h5_index"].values,
        hyper_df["h5_index"].values,
    ])
    all_labels = np.concatenate([
        np.zeros(len(normal_df), dtype=np.int64),
        np.ones(len(hyper_df), dtype=np.int64),
    ])
    all_subject_ids = np.concatenate([
        normal_df["subject_id"].values,
        hyper_df["subject_id"].values,
    ])
    all_study_ids = np.concatenate([
        normal_df["study_id"].values,
        hyper_df["study_id"].values,
    ])
    all_k_values = np.concatenate([
        normal_df["k_value"].values,
        hyper_df["k_value"].values,
    ]).astype(np.float32)
    all_time_diff = np.concatenate([
        normal_df["time_diff"].values,
        hyper_df["time_diff"].values,
    ]).astype(np.float32)

    # 셔플
    perm = np.random.RandomState(42).permutation(len(all_indices))
    all_indices = all_indices[perm]
    all_labels = all_labels[perm]
    all_subject_ids = all_subject_ids[perm]
    all_study_ids = all_study_ids[perm]
    all_k_values = all_k_values[perm]
    all_time_diff = all_time_diff[perm]

    print(f"\n[4/4] HDF5 저장: {OUTPUT_H5}")
    print(f"  Total: {len(all_indices):,} (pos: {all_labels.sum():,}, "
          f"neg: {(all_labels == 0).sum():,}, ratio: 1:{(all_labels == 0).sum() / max(all_labels.sum(), 1):.1f})")

    with h5py.File(OUTPUT_H5, "w") as f:
        f.create_dataset("indices", data=all_indices)
        f.create_dataset("hk_label", data=all_labels)
        f.create_dataset("subject_id", data=all_subject_ids)
        f.create_dataset("study_id", data=all_study_ids)
        f.create_dataset("k_value", data=all_k_values)
        f.create_dataset("time_diff_sec", data=all_time_diff)

    print("Done!")


if __name__ == "__main__":
    main()
