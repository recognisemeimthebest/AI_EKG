"""
PAF 실험용 compact 캐시 생성

ecg_preprocessed.h5 (775K × 5000 × 12, float32, ~173GB)
→ ecg_paf_cache.h5 (523K × 5000 × 2, float16, ~4.9GB)

PAF 데이터셋에 필요한 ECG 인덱스만 추출, Lead I+II만 저장
12-lead → 2-lead 로 6배 읽기 크기 감소 → epoch당 ~3배 속도 향상
"""
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm

ECG_H5 = "g:/AIEKG/ml/data/ecg_preprocessed.h5"
PAF_H5 = "g:/AIEKG/ml/data/ecg_paroxysmal_af.h5"
OUTPUT_H5 = "g:/AIEKG/ml/data/ecg_paf_cache.h5"
CHUNK = 1000  # 배치 단위 읽기


def main():
    print("[1/3] PAF 인덱스 로드...")
    with h5py.File(PAF_H5, "r") as f:
        paf_ecg_indices = f["indices"][:]  # 523K
    print(f"  PAF ECG 수: {len(paf_ecg_indices):,}")

    # 중복 제거 후 정렬 (HDF5 순차 읽기 최적화)
    unique_indices = np.unique(paf_ecg_indices)
    print(f"  고유 ECG 인덱스: {len(unique_indices):,}")

    # 원본 인덱스 → 캐시 인덱스 매핑
    orig_to_cache = {orig: cache for cache, orig in enumerate(unique_indices)}
    cache_ecg_indices = np.array([orig_to_cache[i] for i in paf_ecg_indices], dtype=np.int64)

    est_size = len(unique_indices) * 5000 * 2 * 2 / 1024**3
    print(f"  예상 캐시 크기: {est_size:.1f}GB (float16, 2-lead)")

    print("\n[2/3] Waveform 추출 및 저장...")
    with h5py.File(ECG_H5, "r") as src, h5py.File(OUTPUT_H5, "w") as dst:
        n = len(unique_indices)
        # float16, Lead I(0), Lead II(1)만 저장
        dst.create_dataset(
            "waveform",
            shape=(n, 5000, 2),
            dtype=np.float16,
            chunks=(64, 5000, 2),
            compression="lzf",  # 빠른 압축
        )
        # 인덱스 매핑 저장
        dst.create_dataset("orig_indices", data=unique_indices)
        dst.create_dataset("paf_to_cache", data=cache_ecg_indices)

        for start in tqdm(range(0, n, CHUNK), desc="  copying"):
            end = min(start + CHUNK, n)
            batch_orig = unique_indices[start:end]
            # HDF5는 정렬된 인덱스로 읽어야 빠름
            wf = src["waveform"][batch_orig, :, :2]  # (B, 5000, 2)
            dst["waveform"][start:end] = wf.astype(np.float16)

    print(f"\n[3/3] 완료!")
    import os
    size_gb = os.path.getsize(OUTPUT_H5) / 1024**3
    print(f"  캐시 파일: {OUTPUT_H5}")
    print(f"  실제 크기: {size_gb:.2f}GB")
    print(f"  waveform shape: ({n}, 5000, 2)")


if __name__ == "__main__":
    main()
