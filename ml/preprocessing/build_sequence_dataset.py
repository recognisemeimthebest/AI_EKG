"""
시퀀스 예측 데이터셋 구축: 환자의 연속 Normal ECG → 15일 내 리듬 이상 예측

각 시퀀스: 같은 환자의 연속 Normal ECG 인덱스 (최대 max_seq_len개)
라벨: 마지막 ECG 이후 15일 내 리듬 이상 발생 여부

출력: ecg_sequence_15d_rhythm.h5
  - sequences: (N, max_seq_len) 패딩된 ECG 인덱스 (-1=패딩)
  - seq_lengths: (N,) 실제 시퀀스 길이
  - pred_label: (N,) 0/1
  - days_to_event: (N,) 발생까지 일수
  - subject_id: (N,)
"""
import argparse
import numpy as np
import pandas as pd
import h5py

MACHINE_CSV = "g:/AIEKG/data/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/machine_measurements.csv"
INPUT_H5 = "g:/AIEKG/ml/data/ecg_preprocessed.h5"
RECORD_CSV = "g:/AIEKG/data/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/record_list.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=15)
    parser.add_argument("--max-seq-len", type=int, default=8,
                        help="시퀀스 최대 길이 (최근 N개 ECG 사용)")
    parser.add_argument("--min-seq-len", type=int, default=2,
                        help="시퀀스 최소 길이")
    args = parser.parse_args()

    WINDOW_DAYS = args.window
    MAX_SEQ = args.max_seq_len
    MIN_SEQ = args.min_seq_len
    OUTPUT_H5 = f"g:/AIEKG/ml/data/ecg_sequence_{WINDOW_DAYS}d_rhythm.h5"

    print(f"[1/4] 데이터 로드... (윈도우: {WINDOW_DAYS}일, 시퀀스: {MIN_SEQ}~{MAX_SEQ})")

    # HDF5 메타
    with h5py.File(INPUT_H5, "r") as f:
        subject_ids = f["subject_id"][:]
        study_ids = f["study_id"][:]
        labels = f["label"][:]

    # ECG 시간
    record_list = pd.read_csv(RECORD_CSV, usecols=["subject_id", "study_id", "ecg_time"])
    record_list["ecg_time"] = pd.to_datetime(record_list["ecg_time"])

    # 리듬 이상 study_id
    mm = pd.read_csv(MACHINE_CSV, usecols=["study_id", "report_0", "report_1", "report_2"])
    rhythm_keywords = ["fibrillation", "flutter", "ectopic atrial",
                       "atrial tachycardia", "junctional", "svt",
                       "supraventricular tachycardia"]
    mask = mm["report_0"].str.lower().str.contains("|".join(rhythm_keywords), na=False)
    rhythm_study_ids = set(mm.loc[mask, "study_id"].values)
    print(f"  리듬 이상 ECG: {len(rhythm_study_ids):,}")

    # 품질 문제 ECG 필터링
    quality_keywords = ["warning", "quality", "artifact", "noise", "poor",
                        "uninterpretable", "unable", "technically",
                        "wandering", "tremor", "movement"]
    noisy_study_ids = set()
    for col in ["report_0", "report_1", "report_2"]:
        for kw in quality_keywords:
            m = mm[col].str.lower().str.contains(kw, na=False)
            noisy_study_ids.update(mm.loc[m, "study_id"].values)
    print(f"  노이즈 ECG: {len(noisy_study_ids):,} (제외)")

    # 병합
    h5_df = pd.DataFrame({
        "h5_index": np.arange(len(subject_ids)),
        "subject_id": subject_ids,
        "study_id": study_ids,
        "label": labels,
    })
    h5_df["is_rhythm_abnormal"] = h5_df["study_id"].isin(rhythm_study_ids).astype(int)
    h5_df["is_noisy"] = h5_df["study_id"].isin(noisy_study_ids).astype(int)
    merged = h5_df.merge(record_list, on=["subject_id", "study_id"], how="left")
    merged = merged.sort_values(["subject_id", "ecg_time"]).reset_index(drop=True)

    print(f"  전체 레코드: {len(merged):,}")
    print(f"  전체 환자: {merged['subject_id'].nunique():,}")

    # [2/4] 시퀀스 구축
    print(f"\n[2/4] 시퀀스 구축...")
    sequences = []
    time_gaps_all = []
    seq_lengths = []
    seq_labels = []
    seq_days = []
    seq_sids = []

    for sid, group in merged.groupby("subject_id"):
        rows = group.sort_values("ecg_time")
        times = rows["ecg_time"].values
        labs = rows["label"].values
        h5_idxs = rows["h5_index"].values
        rhythm_flags = rows["is_rhythm_abnormal"].values
        noisy_flags = rows["is_noisy"].values

        # Normal ECG 인덱스만 수집 (노이즈 제외)
        normal_positions = []
        for i in range(len(labs)):
            if labs[i] == 0 and rhythm_flags[i] == 0 and noisy_flags[i] == 0:
                normal_positions.append(i)

        if len(normal_positions) < MIN_SEQ:
            continue

        # 각 Normal ECG 위치에서 시퀀스 생성
        for end_pos_idx in range(MIN_SEQ - 1, len(normal_positions)):
            anchor = normal_positions[end_pos_idx]
            anchor_time = times[anchor]

            # anchor 이전의 Normal ECG를 모아서 시퀀스 구성
            seq_positions = []
            for k in range(end_pos_idx, -1, -1):
                pos = normal_positions[k]
                days_back = (anchor_time - times[pos]) / np.timedelta64(1, "D")
                if days_back > 30:  # 30일 이전은 너무 오래됨
                    break
                seq_positions.append(pos)
                if len(seq_positions) >= MAX_SEQ:
                    break
            seq_positions.reverse()

            if len(seq_positions) < MIN_SEQ:
                continue

            # 라벨: anchor 이후 WINDOW_DAYS 내 리듬 이상 발생?
            found = False
            event_days = np.nan
            for j in range(anchor + 1, len(labs)):
                days = (times[j] - anchor_time) / np.timedelta64(1, "D")
                if days > WINDOW_DAYS:
                    break
                if rhythm_flags[j] == 1:
                    found = True
                    event_days = days
                    break

            if not found:
                # negative: 윈도우 내 follow-up 있어야 확인됨
                has_followup = False
                for j in range(anchor + 1, len(labs)):
                    days = (times[j] - anchor_time) / np.timedelta64(1, "D")
                    if days <= WINDOW_DAYS:
                        has_followup = True
                        break
                    else:
                        break
                if not has_followup:
                    continue

            # 패딩
            seq_indices = [h5_idxs[p] for p in seq_positions]
            padded = [-1] * MAX_SEQ
            for k, idx in enumerate(seq_indices):
                padded[MAX_SEQ - len(seq_indices) + k] = idx  # 오른쪽 정렬

            # 시간 간격 (마지막 ECG 기준, 일 단위)
            time_gaps = [0.0] * MAX_SEQ
            for k, pos in enumerate(seq_positions):
                days_from_anchor = (anchor_time - times[pos]) / np.timedelta64(1, "D")
                time_gaps[MAX_SEQ - len(seq_positions) + k] = days_from_anchor

            sequences.append(padded)
            time_gaps_all.append(time_gaps)
            seq_lengths.append(len(seq_indices))
            seq_labels.append(1 if found else 0)
            seq_days.append(event_days)
            seq_sids.append(sid)

    sequences = np.array(sequences, dtype=np.int64)
    time_gaps_all = np.array(time_gaps_all, dtype=np.float32)
    seq_lengths = np.array(seq_lengths, dtype=np.int32)
    seq_labels = np.array(seq_labels, dtype=np.int64)
    seq_days = np.array(seq_days, dtype=np.float32)
    seq_sids = np.array(seq_sids, dtype=np.int64)

    n_pos = (seq_labels == 1).sum()
    n_neg = (seq_labels == 0).sum()
    print(f"  Positive (리듬 이상): {n_pos:,}")
    print(f"  Negative: {n_neg:,}")
    print(f"  총: {len(seq_labels):,}")
    print(f"  양성 비율: {n_pos / len(seq_labels) * 100:.1f}%")
    print(f"  평균 시퀀스 길이: {seq_lengths.mean():.1f}")
    # 시간 간격 통계 (패딩 제외)
    valid_gaps = time_gaps_all[time_gaps_all > 0]
    if len(valid_gaps) > 0:
        print(f"  시간 간격: 평균 {valid_gaps.mean():.1f}일, 최대 {valid_gaps.max():.1f}일")
    if n_pos > 0:
        print(f"  Positive 평균 발생일: {np.nanmean(seq_days[seq_labels == 1]):.1f}일")

    # [3/4] 저장
    print(f"\n[3/4] 저장: {OUTPUT_H5}")
    with h5py.File(OUTPUT_H5, "w") as f:
        f.create_dataset("sequences", data=sequences)
        f.create_dataset("time_gaps", data=time_gaps_all)
        f.create_dataset("seq_lengths", data=seq_lengths)
        f.create_dataset("pred_label", data=seq_labels)
        f.create_dataset("days_to_event", data=seq_days)
        f.create_dataset("subject_id", data=seq_sids)
        f.attrs["window_days"] = WINDOW_DAYS
        f.attrs["max_seq_len"] = MAX_SEQ
        f.attrs["min_seq_len"] = MIN_SEQ
        f.attrs["padding_value"] = -1
    print("  완료!")


if __name__ == "__main__":
    main()
