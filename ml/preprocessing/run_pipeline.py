"""
ECG 전처리 메인 파이프라인 (병렬 버전)

병렬화 전략:
  - 워커 프로세스: 파형 읽기 + 필터링 + 정규화 (CPU 병렬)
  - 메인 프로세스: HDF5 순차 쓰기 (I/O는 병렬화 불가)

사용법:
  python run_pipeline.py                        # 전체 실행 (12 workers)
  python run_pipeline.py --max-samples 1000     # 테스트
  python run_pipeline.py --workers 8            # 워커 수 조정
"""
import sys
import argparse
import numpy as np
import pandas as pd
import h5py
import wfdb
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# 같은 디렉토리에서 임포트
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    ECG_DATA_DIR, ECG_FILES_DIR, RECORD_LIST_CSV, MACHINE_MEAS_CSV, OUTPUT_DIR,
    DB_CONFIG, N_SAMPLES, N_LEADS, LEAD_NAMES, CLASS_NAMES,
    NUMERIC_FEATURES,
)
from label_mapper import build_label_dataframe
from ecg_filter import preprocess_ecg, check_signal_quality
from patient_features import load_patient_features, load_patient_features_csv


def process_one_record(args):
    """
    단일 레코드 처리 (워커 프로세스에서 실행).
    워커에는 (index, path문자열)만 전달 → 메모리 절약.
    Returns: (index, processed_signal) 또는 (index, None, fail_reason)
    """
    row_idx, record_path = args

    # 파형 로드
    try:
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal
        if signal.shape != (N_SAMPLES, N_LEADS):
            return (row_idx, None, "shape_mismatch")
    except Exception:
        return (row_idx, None, "load_fail")

    # 품질 검사 (필터링 전)
    if not check_signal_quality(signal):
        return (row_idx, None, "quality_fail_pre")

    # 전처리: bandpass → notch → z-score
    processed = preprocess_ecg(signal)

    # 전처리 후 품질 재검사
    if not check_signal_quality(processed):
        return (row_idx, None, "quality_fail_post")

    return (row_idx, processed, "ok")


def main():
    parser = argparse.ArgumentParser(description="ECG 전처리 파이프라인 (병렬)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="처리할 최대 샘플 수 (테스트용)")
    parser.add_argument("--workers", type=int, default=None,
                        help="워커 프로세스 수 (기본: CPU코어-2)")
    parser.add_argument("--use-db", action="store_true",
                        help="DB에서 환자 피처 로드 (기본: CSV)")
    args = parser.parse_args()

    n_workers = args.workers or max(1, cpu_count() - 2)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ============================
    # Step 1: 라벨 + 수치 피처 로드
    # ============================
    print("=" * 60)
    print("[Step 1/5] 라벨 + 수치 피처 로드 중...")
    label_df = build_label_dataframe(str(MACHINE_MEAS_CSV))
    print(f"  유효 레코드: {len(label_df):,}")
    for cls_id, cls_name in CLASS_NAMES.items():
        count = (label_df["label"] == cls_id).sum()
        print(f"    Class {cls_id} ({cls_name}): {count:,}")

    # ============================
    # Step 2: 환자 피처 로드
    # ============================
    print("\n[Step 2/5] 환자 피처 로드 중...")
    if args.use_db:
        try:
            patient_df = load_patient_features(DB_CONFIG)
            print(f"  DB에서 환자 {len(patient_df):,}명 로드")
        except Exception as e:
            print(f"  DB 연결 실패: {e}")
            print("  CSV fallback 시도...")
            patients_csv = Path("g:/AIEKG/data/mimic-iv-3.1/hosp/patients.csv.gz")
            patient_df = load_patient_features_csv(str(patients_csv))
            print(f"  CSV에서 환자 {len(patient_df):,}명 로드")
    else:
        patients_csv = Path("g:/AIEKG/data/mimic-iv-3.1/hosp/patients.csv.gz")
        patient_df = load_patient_features_csv(str(patients_csv))
        print(f"  CSV에서 환자 {len(patient_df):,}명 로드")

    # ============================
    # Step 3: record_list 로드 + 조인
    # ============================
    print("\n[Step 3/5] 레코드 목록 로드 및 데이터 조인 중...")
    record_df = pd.read_csv(str(RECORD_LIST_CSV))
    print(f"  전체 레코드: {len(record_df):,}")

    merged = record_df.merge(label_df, on=["subject_id", "study_id"], how="inner")
    print(f"  라벨 매칭 후: {len(merged):,}")

    merged = merged.merge(patient_df, on="subject_id", how="left")
    print(f"  환자 피처 매칭 후: {len(merged):,}")

    before = len(merged)
    merged = merged.dropna(subset=["age", "gender_code"])
    print(f"  결측 제거 후: {len(merged):,} (제거: {before - len(merged):,})")

    if args.max_samples:
        merged = merged.head(args.max_samples)
        print(f"  테스트 모드: {len(merged):,}개만 처리")

    # ============================
    # Step 4: 병렬 전처리 + HDF5 저장
    # ============================
    print(f"\n[Step 4/5] 병렬 전처리 (워커 {n_workers}개) + HDF5 저장 중...")
    n_total = len(merged)
    output_path = OUTPUT_DIR / "ecg_preprocessed.h5"

    # 워커에는 (index, path)만 전달 → 메모리 절약
    # 메타데이터는 numpy array로 메인 프로세스에서 직접 접근
    merged = merged.reset_index(drop=True)
    paths = [str(ECG_DATA_DIR / p) for p in merged["path"].values]
    tasks = [(i, paths[i]) for i in range(n_total)]

    # 메타데이터를 numpy로 미리 변환 (dict 리스트보다 훨씬 가벼움)
    meta_labels = merged["label"].values.astype(np.int8)
    meta_numeric = merged[["rr_interval", "qrs_duration", "p_duration", "qrs_axis"]].fillna(0).values.astype(np.float32)
    meta_patient = merged[["age", "gender_code"]].values.astype(np.float32)
    meta_subject = merged["subject_id"].values.astype(np.int64)
    meta_study = merged["study_id"].values.astype(np.int64)
    del merged, paths  # DataFrame 메모리 해제

    stats = {"success": 0, "load_fail": 0, "quality_fail": 0, "shape_mismatch": 0}

    with h5py.File(str(output_path), "w") as hf:
        ds_waveform = hf.create_dataset(
            "waveform", shape=(n_total, N_SAMPLES, N_LEADS),
            dtype="float32", chunks=(1, N_SAMPLES, N_LEADS),
            maxshape=(None, N_SAMPLES, N_LEADS),
        )  # 압축 제거: gzip이 메모리 병목 → 파일 크기↑ 대신 메모리 안정
        ds_label = hf.create_dataset(
            "label", shape=(n_total,), dtype="int8",
            maxshape=(None,), chunks=(min(n_total, 1000),)
        )
        ds_numeric = hf.create_dataset(
            "numeric_features", shape=(n_total, len(NUMERIC_FEATURES)),
            dtype="float32", maxshape=(None, len(NUMERIC_FEATURES)),
            chunks=(min(n_total, 1000), len(NUMERIC_FEATURES))
        )
        ds_patient = hf.create_dataset(
            "patient_features", shape=(n_total, 2),
            dtype="float32", maxshape=(None, 2),
            chunks=(min(n_total, 1000), 2)
        )
        ds_subject = hf.create_dataset(
            "subject_id", shape=(n_total,), dtype="int64",
            maxshape=(None,), chunks=(min(n_total, 1000),)
        )
        ds_study = hf.create_dataset(
            "study_id", shape=(n_total,), dtype="int64",
            maxshape=(None,), chunks=(min(n_total, 1000),)
        )

        idx = 0
        # 병렬 처리: imap_unordered로 완료되는 순서대로 수집
        with Pool(processes=n_workers) as pool:
            results = pool.imap_unordered(process_one_record, tasks, chunksize=4)

            for result in tqdm(results, total=n_total, desc=f"전처리 (x{n_workers})"):
                if len(result) == 3 and result[1] is None:
                    # 실패
                    row_idx, _, reason = result
                    if reason == "load_fail":
                        stats["load_fail"] += 1
                    elif reason == "shape_mismatch":
                        stats["shape_mismatch"] += 1
                    else:
                        stats["quality_fail"] += 1
                    continue

                # 성공 - HDF5에 저장
                row_idx, processed, _ = result
                ds_waveform[idx] = processed
                ds_label[idx] = meta_labels[row_idx]
                ds_numeric[idx] = meta_numeric[row_idx]
                ds_patient[idx] = meta_patient[row_idx]
                ds_subject[idx] = meta_subject[row_idx]
                ds_study[idx] = meta_study[row_idx]
                idx += 1
                stats["success"] += 1

        # 실제 크기로 리사이즈
        for key in hf.keys():
            shape = list(hf[key].shape)
            shape[0] = idx
            hf[key].resize(shape)

        # 메타데이터 저장
        hf.attrs["n_samples"] = idx
        hf.attrs["sample_rate"] = 500
        hf.attrs["duration_sec"] = 10
        hf.attrs["n_leads"] = N_LEADS
        hf.attrs["lead_names"] = LEAD_NAMES
        hf.attrs["class_names"] = list(CLASS_NAMES.values())
        hf.attrs["numeric_feature_names"] = NUMERIC_FEATURES
        hf.attrs["patient_feature_names"] = ["age", "gender_code"]

    # ============================
    # Step 5: 결과 요약
    # ============================
    print(f"\n[Step 5/5] 전처리 완료!")
    print("=" * 60)
    print(f"  워커 수:       {n_workers:>10}")
    print(f"  처리 대상:     {n_total:>10,}")
    print(f"  성공:          {stats['success']:>10,}")
    print(f"  로드 실패:     {stats['load_fail']:>10,}")
    print(f"  품질 불량:     {stats['quality_fail']:>10,}")
    print(f"  형태 불일치:   {stats['shape_mismatch']:>10,}")
    print(f"  저장 위치:     {output_path}")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    file_size_gb = file_size_mb / 1024
    print(f"  파일 크기:     {file_size_gb:>10.1f} GB ({file_size_mb:,.0f} MB)")

    with h5py.File(str(output_path), "r") as hf:
        labels = hf["label"][:]
        print(f"\n  최종 클래스 분포:")
        for cls_id, cls_name in CLASS_NAMES.items():
            count = np.sum(labels == cls_id)
            pct = count / len(labels) * 100 if len(labels) > 0 else 0
            print(f"    {cls_id} ({cls_name:<8}): {count:>8,} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
