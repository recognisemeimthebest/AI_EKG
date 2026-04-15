"""
시계열 예측 데이터셋 구축: Normal ECG → 30일 내 AFib 발생 예측

입력: 기존 ecg_preprocessed.h5 + record_list.csv
출력: ecg_prediction_30d.h5
  - indices: 기존 HDF5의 인덱스 (waveform 재사용)
  - pred_label: 0=유지, 1=30일 내 AFib 발생
  - days_to_event: AFib까지 일수 (없으면 NaN)
  - subject_id, study_id
"""
import argparse
import numpy as np
import pandas as pd
import h5py
from pathlib import Path

INPUT_H5 = "g:/AIEKG/ml/data/ecg_preprocessed.h5"
RECORD_CSV = "g:/AIEKG/data/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/record_list.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=30, help="예측 윈도우 (일)")
    parser.add_argument("--target", type=str, default="afib",
                        choices=["afib", "rhythm"],
                        help="예측 대상: afib(AFib만) 또는 rhythm(리듬이상그룹)")
    args = parser.parse_args()
    WINDOW_DAYS = args.window
    TARGET = args.target
    OUTPUT_H5 = f"g:/AIEKG/ml/data/ecg_prediction_{WINDOW_DAYS}d{'_rhythm' if TARGET == 'rhythm' else ''}.h5"

    print(f"[1/3] 데이터 로드... (윈도우: {WINDOW_DAYS}일, 대상: {TARGET})")
    record_list = pd.read_csv(RECORD_CSV, usecols=["subject_id", "study_id", "ecg_time"])
    record_list["ecg_time"] = pd.to_datetime(record_list["ecg_time"])

    with h5py.File(INPUT_H5, "r") as f:
        subject_ids = f["subject_id"][:]
        study_ids = f["study_id"][:]
        labels = f["label"][:]

    # rhythm 모드: machine_measurements에서 리듬 이상 study_id 추출
    rhythm_study_ids = set()
    if TARGET == "rhythm":
        MACHINE_CSV = "g:/AIEKG/data/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/machine_measurements.csv"
        mm = pd.read_csv(MACHINE_CSV, usecols=["study_id", "report_0"])
        rhythm_keywords = ["fibrillation", "flutter", "ectopic atrial",
                           "atrial tachycardia", "junctional", "svt",
                           "supraventricular tachycardia"]
        mask = mm["report_0"].str.lower().str.contains("|".join(rhythm_keywords), na=False)
        rhythm_study_ids = set(mm.loc[mask, "study_id"].values)
        print(f"  리듬 이상 ECG: {len(rhythm_study_ids):,}")

    h5_df = pd.DataFrame({
        "h5_index": np.arange(len(subject_ids)),
        "subject_id": subject_ids,
        "study_id": study_ids,
        "label": labels,
    })

    # rhythm 모드: label 재정의 (리듬 이상이면 1, 아니면 원래 label 유지)
    if TARGET == "rhythm":
        h5_df["is_rhythm_abnormal"] = h5_df["study_id"].isin(rhythm_study_ids).astype(int)
    merged = h5_df.merge(record_list, on=["subject_id", "study_id"], how="left")
    merged = merged.sort_values(["subject_id", "ecg_time"]).reset_index(drop=True)

    print(f"  전체 레코드: {len(merged):,}")
    print(f"  전체 환자: {merged['subject_id'].nunique():,}")

    print(f"\n[2/3] {WINDOW_DAYS}일 윈도우 라벨링...")
    result_indices = []
    result_labels = []
    result_days = []
    result_sids = []
    result_studyids = []

    for sid, group in merged.groupby("subject_id"):
        if len(group) < 2:
            continue

        rows = group.sort_values("ecg_time")
        times = rows["ecg_time"].values
        labs = rows["label"].values
        h5_idxs = rows["h5_index"].values
        sids = rows["subject_id"].values
        stuids = rows["study_id"].values

        # rhythm 모드면 is_rhythm_abnormal 컬럼 사용
        if TARGET == "rhythm":
            rhythm_flags = rows["is_rhythm_abnormal"].values

        for i in range(len(labs)):
            if labs[i] != 0:  # Normal만 대상
                continue
            if TARGET == "rhythm" and rhythm_flags[i] == 1:
                continue  # 현재 ECG가 리듬 이상이면 스킵

            found_positive = False
            for j in range(i + 1, len(labs)):
                days = (times[j] - times[i]) / np.timedelta64(1, "D")
                if days > WINDOW_DAYS:
                    break

                is_target = False
                if TARGET == "afib":
                    is_target = (labs[j] == 1)
                elif TARGET == "rhythm":
                    is_target = (rhythm_flags[j] == 1)

                if is_target:
                    result_indices.append(h5_idxs[i])
                    result_labels.append(1)
                    result_days.append(days)
                    result_sids.append(sids[i])
                    result_studyids.append(stuids[i])
                    found_positive = True
                    break

            if not found_positive:
                has_followup = False
                for j in range(i + 1, len(labs)):
                    days = (times[j] - times[i]) / np.timedelta64(1, "D")
                    if days <= WINDOW_DAYS:
                        has_followup = True
                        break
                    else:
                        break

                if has_followup:
                    result_indices.append(h5_idxs[i])
                    result_labels.append(0)
                    result_days.append(np.nan)
                    result_sids.append(sids[i])
                    result_studyids.append(stuids[i])

    result_indices = np.array(result_indices, dtype=np.int64)
    result_labels = np.array(result_labels, dtype=np.int64)
    result_days = np.array(result_days, dtype=np.float32)
    result_sids = np.array(result_sids, dtype=np.int64)
    result_studyids = np.array(result_studyids, dtype=np.int64)

    n_pos = (result_labels == 1).sum()
    n_neg = (result_labels == 0).sum()
    target_name = "AFib" if TARGET == "afib" else "Rhythm Abnormal"
    print(f"  Positive ({WINDOW_DAYS}일 내 {target_name}): {n_pos:,}")
    print(f"  Negative (유지): {n_neg:,}")
    print(f"  총: {len(result_labels):,}")
    print(f"  비율: {n_pos / len(result_labels) * 100:.1f}%")
    if n_pos > 0:
        print(f"  Positive 평균 발생일: {np.nanmean(result_days[result_labels == 1]):.1f}일")

    print(f"\n[3/3] HDF5 저장: {OUTPUT_H5}")
    with h5py.File(OUTPUT_H5, "w") as f:
        f.create_dataset("indices", data=result_indices)
        f.create_dataset("pred_label", data=result_labels)
        f.create_dataset("days_to_event", data=result_days)
        f.create_dataset("subject_id", data=result_sids)
        f.create_dataset("study_id", data=result_studyids)
        f.attrs["window_days"] = WINDOW_DAYS
        f.attrs["source_h5"] = INPUT_H5
        f.attrs["positive_label"] = f"{target_name} within {WINDOW_DAYS} days"
        f.attrs["negative_label"] = f"No {target_name} within {WINDOW_DAYS} days"

    print("  완료!")


if __name__ == "__main__":
    main()
