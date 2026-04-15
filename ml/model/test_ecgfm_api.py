"""
ECG-FM API 검증 스크립트
- import, 모델 로딩, forward pass 모양 확인
- 풀 학습 전에 먼저 실행해서 인터페이스 확인

WSL2 실행:
  source /mnt/g/AIEKG/venv_wsl/bin/activate
  cd /mnt/g/AIEKG/ml/model
  python test_ecgfm_api.py
"""

import sys
import torch
from pathlib import Path

CKPT = "/mnt/g/AIEKG/ml/checkpoints/ecg-fm/mimic_iv_ecg_finetuned.pt"

print("=" * 60)
print("ECG-FM API 검증")
print("=" * 60)

# ── 1. Import ─────────────────────────────────────────────────
print("\n[1] fairseq-signals import...")
try:
    from fairseq_signals.models import build_model_from_checkpoint
    print("  OK: build_model_from_checkpoint 임포트 성공")
except ImportError as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# ── 2. 모델 로딩 ──────────────────────────────────────────────
print(f"\n[2] 모델 로딩: {CKPT}")
try:
    result = build_model_from_checkpoint(CKPT)
    if isinstance(result, tuple):
        model = result[0]
        print(f"  OK: build_model_from_checkpoint 반환 튜플, len={len(result)}")
    else:
        model = result
        print(f"  OK: build_model_from_checkpoint 반환 단일 모델")
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  파라미터 수: {n_params:,}")
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ── 3. 디바이스 ───────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[3] Device: {device}")
model = model.to(device)

# ── 4. Forward pass 테스트 ────────────────────────────────────
print("\n[4] Forward pass 테스트 (batch=2, 12-lead, 2500 samples)...")
B = 2
dummy = torch.randn(B, 12, 2500, device=device)

with torch.no_grad():
    # 방법 1: extract_features
    print("  방법 1: model.extract_features(source=..., padding_mask=None)")
    try:
        out = model.extract_features(source=dummy, padding_mask=None)
        if isinstance(out, dict):
            print(f"  OK: 반환 dict, keys={list(out.keys())}")
            for k, v in out.items():
                if isinstance(v, torch.Tensor):
                    print(f"    [{k}] shape={v.shape}")
        elif isinstance(out, tuple):
            print(f"  OK: 반환 tuple, len={len(out)}")
            for i, v in enumerate(out):
                if isinstance(v, torch.Tensor):
                    print(f"    [{i}] shape={v.shape}")
        else:
            print(f"  OK: 반환 {type(out)}")
    except Exception as e:
        print(f"  FAIL: {e}")

    # 방법 2: forward
    print("  방법 2: model(source=...)")
    try:
        out2 = model(source=dummy, padding_mask=None)
        if isinstance(out2, dict):
            print(f"  OK: 반환 dict, keys={list(out2.keys())}")
            for k, v in out2.items():
                if isinstance(v, torch.Tensor):
                    print(f"    [{k}] shape={v.shape}")
        elif isinstance(out2, tuple):
            print(f"  OK: 반환 tuple, len={len(out2)}")
            for i, v in enumerate(out2):
                if isinstance(v, torch.Tensor):
                    print(f"    [{i}] shape={v.shape}")
    except Exception as e:
        print(f"  FAIL: {e}")

# ── 5. 3-lead zero-pad 테스트 ─────────────────────────────────
print("\n[5] 3-lead zero-pad 패턴 테스트...")
seg_3lead = torch.randn(B, 3, 2500, device=device)
padded_12 = torch.zeros(B, 12, 2500, device=device)
padded_12[:, :3, :] = seg_3lead
print(f"  zero-padded shape: {padded_12.shape}")
print(f"  leads 0-2 nonzero: {(padded_12[:, :3, :] != 0).sum().item()}")
print(f"  leads 3-11 all zero: {(padded_12[:, 3:, :] == 0).all().item()}")

# ── 6. 메모리 체크 ───────────────────────────────────────────
if device.type == "cuda":
    print(f"\n[6] GPU 메모리 사용: {torch.cuda.memory_allocated()/1e9:.2f}GB / "
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

print("\n" + "=" * 60)
print("검증 완료 - 위 결과를 바탕으로 train_ecgfm.py 인터페이스 확인")
print("extract_features 반환 dict의 'x' 키가 (B, T', 768)이면 그대로 사용 가능")
print("=" * 60)
