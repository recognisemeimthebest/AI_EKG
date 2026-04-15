"""
ECG-FM 피처 사전 추출 (1회 실행)

시퀀스에서 참조되는 ECG (119K) 만 추출하여 768-dim 벡터로 저장.
fp16 + 배치 처리로 속도 최적화.

출력: /mnt/g/AIEKG/ml/data/ecgfm_features.h5
  - features:    (N, 768)  float16
  - ecg_indices: (N,)      int32   — ecg_preprocessed.h5 상의 원래 인덱스

WSL2 실행:
  source /mnt/g/AIEKG/venv_wsl/bin/activate
  cd /mnt/g/AIEKG/ml/model
  python extract_ecgfm_features.py
"""

import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from fairseq_signals.models import build_model_from_checkpoint

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
ECG_H5    = "/mnt/g/AIEKG/ml/data/ecg_preprocessed.h5"
SEQ_H5    = "/mnt/g/AIEKG/ml/data/ecg_sequence_15d_rhythm.h5"
CKPT      = "/mnt/g/AIEKG/ml/checkpoints/ecg-fm/mimic_iv_ecg_finetuned.pt"
OUT_H5    = "/mnt/g/AIEKG/ml/data/ecgfm_features.h5"
BATCH     = 32   # fp16 기준 RTX 4070 Ti SUPER에서 OOM 없이 안전한 크기


def get_required_indices(seq_h5_path):
    """시퀀스에서 실제 참조되는 ECG 인덱스 반환"""
    with h5py.File(seq_h5_path, "r") as f:
        seqs = f["sequences"][:]
    indices = np.unique(seqs[seqs >= 0]).astype(np.int32)
    print(f"  시퀀스에서 참조되는 ECG: {len(indices):,}개")
    return indices


def encode_batch(model, waveforms_np, device):
    """
    waveforms_np: (B, 5000, 12) — ecg_preprocessed.h5의 원래 형식
    반환: (B, 768) float16
    """
    B = waveforms_np.shape[0]

    # 3-lead 구성: I=ch0, II=ch1, III=ch1-ch0
    lead_i   = waveforms_np[:, :, 0]   # (B, 5000)
    lead_ii  = waveforms_np[:, :, 1]
    lead_iii = lead_ii - lead_i         # Einthoven

    # (B, 5000, 3) — float32
    ecg3 = np.stack([lead_i, lead_ii, lead_iii], axis=2).astype(np.float32)

    # 2 세그먼트 분리
    seg1 = ecg3[:, :2500, :]    # (B, 2500, 3)
    seg2 = ecg3[:, 2500:, :]    # (B, 2500, 3)

    def encode_seg(seg_np):
        # zero-pad: (B, 2500, 3) → (B, 12, 2500)
        padded = torch.zeros(B, 12, 2500, dtype=torch.float16, device=device)
        t = torch.from_numpy(seg_np).to(device=device, dtype=torch.float16)
        padded[:, :3, :] = t.transpose(1, 2)   # I,II,III → positions 0,1,2

        with torch.no_grad():
            result = model.extract_features(
                source=padded.float(),   # ECG-FM은 float32 내부 연산
                padding_mask=None
            )
        x = result["encoder_out"]  # (B, T', 768)
        return x.mean(dim=1).half()  # (B, 768) float16

    f1 = encode_seg(seg1)
    f2 = encode_seg(seg2)
    feat = ((f1 + f2) / 2.0).cpu().numpy()   # (B, 768) float16
    return feat


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {name} ({mem:.1f}GB)")

    # ── 1. 필요한 ECG 인덱스 수집 ────────────────────────────────────────────
    print("\n[1/4] 필요한 ECG 인덱스 수집...")
    indices = get_required_indices(SEQ_H5)
    N = len(indices)

    # ── 2. ECG-FM 로딩 ───────────────────────────────────────────────────────
    print(f"\n[2/4] ECG-FM 로딩: {CKPT}")
    result = build_model_from_checkpoint(CKPT)
    model  = result[0] if isinstance(result, tuple) else result
    model  = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  파라미터: {n_params:,}")

    # ── 3. 배치 추출 ─────────────────────────────────────────────────────────
    print(f"\n[3/4] 피처 추출 (batch={BATCH}, fp16)...")
    print(f"  총 {N:,}개 ECG, {(N + BATCH - 1) // BATCH}개 배치")

    features_all = np.zeros((N, 768), dtype=np.float16)
    t0 = time.time()

    with h5py.File(ECG_H5, "r") as ecg_f:
        for start in tqdm(range(0, N, BATCH), desc="  추출 중"):
            end         = min(start + BATCH, N)
            batch_idx   = indices[start:end]
            waveforms   = ecg_f["waveform"][batch_idx]  # (B, 5000, 12)
            features    = encode_batch(model, waveforms, device)
            features_all[start:end] = features

            if (start // BATCH) % 100 == 0 and start > 0:
                elapsed = time.time() - t0
                done    = (start + BATCH) / N
                eta     = elapsed / done * (1 - done) / 60
                print(f"\r  진행: {done:.1%}  경과: {elapsed/60:.1f}분  남은 시간: {eta:.0f}분",
                      end="", flush=True)

    elapsed = time.time() - t0
    print(f"\n  완료! 총 소요: {elapsed/60:.1f}분")

    # ── 4. HDF5 저장 ─────────────────────────────────────────────────────────
    print(f"\n[4/4] 저장: {OUT_H5}")
    with h5py.File(OUT_H5, "w") as f:
        f.create_dataset("features",    data=features_all, compression="gzip",
                         chunks=(min(1000, N), 768))
        f.create_dataset("ecg_indices", data=indices,      compression="gzip")
        f.attrs["n_ecg"]   = N
        f.attrs["feat_dim"] = 768
        f.attrs["model"]   = "mimic_iv_ecg_finetuned (ECG-FM)"
        f.attrs["dtype"]   = "float16"
    print(f"  저장 완료: features={features_all.shape}, dtype=float16")
    size_gb = Path(OUT_H5).stat().st_size / 1e9
    print(f"  파일 크기: {size_gb:.2f}GB")
    print("\n완료! 다음 단계: python train_ecgfm_head.py")


if __name__ == "__main__":
    main()
