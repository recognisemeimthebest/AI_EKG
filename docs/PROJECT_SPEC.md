# AIEKG — AI EKG 측정 장치 프로젝트 기획서 요약

> 전체 기획서: `docs/project-plan-v1.0.md` (v3.1, 2026-04-07)
> 이 파일은 새 세션 브리핑용 요약본이다.

---

## 프로젝트 한 줄 요약

ESP32 + ADS1292R 기반 3리드 EKG 측정 장치 + Raspberry Pi 5 AI 추론 서버 + 5인치 LCD 대시보드. 웰니스 기기 포지셔닝 (의료기기 인증 없음).

---

## 기술 스택

| 레이어 | 선택 |
|--------|------|
| 하드웨어 | ESP32, ADS1292R, 3전극(RA/LA/RL), 5인치 SPI LCD |
| 통신 | BLE → Raspberry Pi 5 |
| 펌웨어 | ESP-IDF / PlatformIO, JSF++ 원칙 |
| AI 서버 | Raspberry Pi 5, PyTorch → ONNX |
| DB | PostgreSQL |
| 데이터 | MIMIC-IV ECG 775,367건, 3-lead (I, II, computed III) |

---

## AI 기능 3종 현황

| # | 기능 | 상태 | 성능 |
|---|------|------|------|
| 1 | 부정맥 분류 (Normal/AFib/Other) | ✅ 완료 | Accuracy 90.6% |
| 2 | 발작성 AF 감지 (정상 리듬에서 숨겨진 AF 스크리닝) | ✅ 완료 | AUROC 0.8240 |
| 3 | AFib 예측 (15일 내 리듬 이상 예측, 임상정보 결합) | 🔄 개선 중 | 현재 AUROC 0.7355 → 목표 0.80+ |

---

## 현재 단계

**Phase 4 (ML 모델 개발)** 진행 중. Phase 1~3 (하드웨어/펌웨어)은 하드웨어 부품 도착 대기 중 보류.

**다음 우선 작업**:
1. [최우선] AFib 예측 모델 개선 — 임상정보 추출 + 결합 아키텍처, AUROC 0.80+ 목표
2. 모델 3종 ONNX 변환 + Raspberry Pi 5 추론 벤치마크

---

## 핵심 제약사항 (절대 잊지 말 것)

- **3-lead 한계**: Lead I, II만 측정, Lead III는 계산값 (II - I). 12-lead 논문 대비 성능 하락 2~4% 감수.
- **규제**: 웰니스 기기 — "병원 방문 권장" 수준만 가능, 구체적 진단/치료 지시 불가.
- **JSF++ 원칙**: 펌웨어 코드에서 동적 메모리 할당 금지, 재귀 금지.
- **BLE 통신**: Wi-Fi/MQTT에서 BLE로 변경됨 (2026-04-06 결정).

---

## 실패 사례 (반드시 참고)

- CNN-TCN Transfer Learning → 발작성 AF 감지 AUROC 0.68 정체 → **기각**
  - 원인: ECG 도메인 특화 pre-training 없이 TL 시도 → 수렴 안 됨
  - 해결: ResNet-34 from scratch → AUROC 0.82
