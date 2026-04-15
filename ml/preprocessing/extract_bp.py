"""
ECG 레코드별 혈압(SBP/DBP) 매칭 스크립트

소스:
  1. ICU chartevents (±6시간, 우선)
  2. OMR (±3일, 폴백)

출력: 기존 HDF5에 bp_features(N, 2) 데이터셋 추가 [SBP, DBP]
매칭 안 되면 NaN → dataset.py에서 0으로 처리
"""
import argparse
import time
import numpy as np
import pandas as pd
import h5py
import psycopg2
from config import DB_CONFIG, ECG_DATA_DIR, RECORD_LIST_CSV

# ICU itemid: Non-Invasive BP 우선, Arterial BP 폴백
SBP_ITEMS = [220179, 220050]  # NIBP systolic, ABP systolic
DBP_ITEMS = [220180, 220051]  # NIBP diastolic, ABP diastolic


def load_ecg_times(h5_path):
    """HDF5에서 subject_id, study_id를 읽고 record_list에서 ecg_time 매칭"""
    with h5py.File(h5_path, "r") as f:
        subject_ids = f["subject_id"][:]
        study_ids = f["study_id"][:]

    record_list = pd.read_csv(RECORD_LIST_CSV, usecols=["subject_id", "study_id", "ecg_time"])
    record_list["ecg_time"] = pd.to_datetime(record_list["ecg_time"])

    # study_id 기반 매칭
    ecg_df = pd.DataFrame({"subject_id": subject_ids, "study_id": study_ids})
    ecg_df = ecg_df.merge(record_list, on=["subject_id", "study_id"], how="left")

    return ecg_df


def fetch_icu_bp(conn, patient_ids):
    """ICU chartevents에서 SBP/DBP를 배치로 로드"""
    all_items = tuple(SBP_ITEMS + DBP_ITEMS)
    batch_size = 5000
    chunks = []

    for i in range(0, len(patient_ids), batch_size):
        batch = patient_ids[i:i + batch_size]
        cur = conn.cursor()
        cur.execute("""
            SELECT subject_id, charttime, itemid, valuenum
            FROM mimiciv_icu.chartevents
            WHERE itemid IN %s
            AND subject_id = ANY(%s)
            AND valuenum IS NOT NULL
            AND valuenum > 0 AND valuenum < 300
        """, (all_items, batch))
        rows = cur.fetchall()
        cur.close()
        if rows:
            chunk = pd.DataFrame(rows, columns=["subject_id", "charttime", "itemid", "valuenum"])
            chunks.append(chunk)
        print(f"    ICU 배치 {i//batch_size+1}/{(len(patient_ids)+batch_size-1)//batch_size}: {len(rows):,}건")

    if not chunks:
        return pd.DataFrame(columns=["subject_id", "charttime", "itemid", "valuenum", "is_sbp"])

    df = pd.concat(chunks, ignore_index=True)
    df["charttime"] = pd.to_datetime(df["charttime"])
    df["is_sbp"] = df["itemid"].isin(SBP_ITEMS)
    return df


def fetch_omr_bp(conn, patient_ids):
    """OMR에서 Blood Pressure를 배치로 로드 → SBP/DBP 분리"""
    batch_size = 10000
    chunks = []

    for i in range(0, len(patient_ids), batch_size):
        batch = patient_ids[i:i + batch_size]
        cur = conn.cursor()
        cur.execute("""
            SELECT subject_id, chartdate, result_value
            FROM mimiciv_hosp.omr
            WHERE result_name = 'Blood Pressure'
            AND subject_id = ANY(%s)
            AND result_value IS NOT NULL
        """, (batch,))
        rows = cur.fetchall()
        cur.close()
        if rows:
            chunk = pd.DataFrame(rows, columns=["subject_id", "chartdate", "result_value"])
            chunks.append(chunk)
        print(f"    OMR 배치 {i//batch_size+1}/{(len(patient_ids)+batch_size-1)//batch_size}: {len(rows):,}건")

    if not chunks:
        return pd.DataFrame(columns=["subject_id", "chartdate", "sbp", "dbp"])

    df = pd.concat(chunks, ignore_index=True)
    df["chartdate"] = pd.to_datetime(df["chartdate"])

    # "120/80" → SBP, DBP
    def parse_bp(val):
        try:
            parts = str(val).split("/")
            return float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            return np.nan, np.nan

    bp_parsed = df["result_value"].apply(parse_bp)
    df["sbp"] = bp_parsed.apply(lambda x: x[0])
    df["dbp"] = bp_parsed.apply(lambda x: x[1])
    df = df.dropna(subset=["sbp", "dbp"])
    df = df[(df["sbp"] > 0) & (df["sbp"] < 300) & (df["dbp"] > 0) & (df["dbp"] < 300)]

    return df


def build_lookup(icu_df, omr_df):
    """환자별 딕셔너리로 그룹핑하여 검색 속도 향상"""
    icu_lookup = {}
    for sid, group in icu_df.groupby("subject_id"):
        sbp = group[group["is_sbp"]][["charttime", "valuenum"]].values
        dbp = group[~group["is_sbp"]][["charttime", "valuenum"]].values
        icu_lookup[sid] = (sbp, dbp)

    omr_lookup = {}
    for sid, group in omr_df.groupby("subject_id"):
        omr_lookup[sid] = group[["chartdate", "sbp", "dbp"]].values

    # 원본 DataFrame 메모리 해제
    return icu_lookup, omr_lookup


def match_bp_for_batch(ecg_batch, icu_lookup, omr_lookup):
    """ECG 배치에 대해 가장 가까운 BP 매칭"""
    results = np.full((len(ecg_batch), 2), np.nan, dtype=np.float32)
    six_hours = np.timedelta64(6, "h")
    three_days = np.timedelta64(3, "D")

    for i, row in enumerate(ecg_batch.itertuples()):
        sid = row.subject_id
        ecg_dt = row.ecg_time
        if pd.isna(ecg_dt):
            continue

        ecg_np = np.datetime64(ecg_dt)

        # 1. ICU ±6시간 매칭 (우선)
        if sid in icu_lookup:
            sbp_arr, dbp_arr = icu_lookup[sid]

            if len(sbp_arr) > 0:
                diffs = np.abs(sbp_arr[:, 0].astype("datetime64[ns]") - ecg_np)
                min_idx = np.argmin(diffs)
                if diffs[min_idx] <= six_hours:
                    results[i, 0] = float(sbp_arr[min_idx, 1])

            if len(dbp_arr) > 0:
                diffs = np.abs(dbp_arr[:, 0].astype("datetime64[ns]") - ecg_np)
                min_idx = np.argmin(diffs)
                if diffs[min_idx] <= six_hours:
                    results[i, 1] = float(dbp_arr[min_idx, 1])

            if not np.isnan(results[i, 0]) and not np.isnan(results[i, 1]):
                continue

        # 2. OMR ±3일 폴백
        if sid in omr_lookup:
            omr_arr = omr_lookup[sid]
            diffs = np.abs(omr_arr[:, 0].astype("datetime64[ns]") - ecg_np)
            min_idx = np.argmin(diffs)
            if diffs[min_idx] <= three_days:
                if np.isnan(results[i, 0]):
                    results[i, 0] = float(omr_arr[min_idx, 1])
                if np.isnan(results[i, 1]):
                    results[i, 1] = float(omr_arr[min_idx, 2])

    return results


def main():
    parser = argparse.ArgumentParser(description="ECG에 혈압 피처 매칭")
    parser.add_argument("--h5", default="g:/AIEKG/ml/data/ecg_preprocessed.h5")
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args()

    print("[1/4] ECG 시간 정보 로드...")
    ecg_df = load_ecg_times(args.h5)
    n_total = len(ecg_df)
    print(f"  ECG 레코드: {n_total:,}")

    unique_patients = [int(x) for x in ecg_df["subject_id"].unique()]
    print(f"  고유 환자: {len(unique_patients):,}")

    print("\n[2/4] DB에서 바이탈사인 로드...")
    conn = psycopg2.connect(**DB_CONFIG)

    t0 = time.time()
    print("  ICU chartevents 로드 중...")
    icu_df = fetch_icu_bp(conn, unique_patients)
    print(f"  ICU BP 레코드: {len(icu_df):,} ({time.time()-t0:.0f}s)")

    t0 = time.time()
    print("  OMR 로드 중...")
    omr_df = fetch_omr_bp(conn, unique_patients)
    print(f"  OMR BP 레코드: {len(omr_df):,} ({time.time()-t0:.0f}s)")
    conn.close()

    print("\n[3/4] ECG-BP 매칭 중...")
    print("  환자별 인덱스 빌드 중...")
    icu_lookup, omr_lookup = build_lookup(icu_df, omr_df)
    del icu_df, omr_df  # 원본 DataFrame 메모리 해제
    import gc; gc.collect()
    print(f"  ICU 환자: {len(icu_lookup):,}, OMR 환자: {len(omr_lookup):,}")

    bp_features = np.full((n_total, 2), np.nan, dtype=np.float32)

    n_batches = (n_total + args.batch_size - 1) // args.batch_size
    matched = 0
    t0 = time.time()

    for batch_idx in range(n_batches):
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, n_total)
        batch = ecg_df.iloc[start:end]

        batch_result = match_bp_for_batch(batch, icu_lookup, omr_lookup)
        bp_features[start:end] = batch_result

        batch_matched = np.sum(~np.isnan(batch_result[:, 0]))
        matched += batch_matched

        if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
            elapsed = time.time() - t0
            pct = end / n_total * 100
            print(f"  {end:,}/{n_total:,} ({pct:.1f}%) "
                  f"매칭: {matched:,} ({matched/end*100:.1f}%) "
                  f"경과: {elapsed:.0f}s")

    print(f"\n  최종 매칭률: {matched:,}/{n_total:,} ({matched/n_total*100:.1f}%)")
    print(f"  SBP 평균: {np.nanmean(bp_features[:,0]):.1f} mmHg")
    print(f"  DBP 평균: {np.nanmean(bp_features[:,1]):.1f} mmHg")

    print("\n[4/4] HDF5에 bp_features 저장...")
    with h5py.File(args.h5, "a") as f:
        if "bp_features" in f:
            del f["bp_features"]
        f.create_dataset("bp_features", data=bp_features)
        f.attrs["bp_feature_names"] = ["sbp", "dbp"]

    print("  완료!")


if __name__ == "__main__":
    main()
