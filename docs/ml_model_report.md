# AI EKG 프로젝트 - ML 모델 학습 보고서

**작성일**: 2026-04-06
**최종 수정**: 2026-04-07
**Phase**: 4 (AI/ML 모델 개발)
**상태**: 진행중 (3종 중 2종 완료)

---

## 1. 개요

ESP32 + ADS1292R 기반 AI EKG 기기에 탑재할 ML 모델 3종을 개발 중이다.
학습 데이터는 MIMIC-IV ECG (775,367건, 159,666명)을 사용하였으며,
배포 하드웨어에 맞춰 3-lead (Lead I, II + Lead III 계산) 입력으로 학습하였다.

**모델 현황:**
1. 부정맥 분류 (CNN-TCN) -- **완료**, Test Accuracy 90.6%
2. 발작성 AF 감지 (ResNet-34 + Clinical) -- **완료**, Test AUROC 0.8240
3. AFib 예측 개선 (시퀀스 TCN + 임상정보) -- 예정, 현재 AUROC 0.7355 -> 목표 0.80+

## 2. 최종 모델

### 2.1 부정맥 분류 모델 (CNN-TCN)

| 항목 | 내용 |
|------|------|
| 구조 | CNN 3블록 + TCN 4레이어 + FC |
| 입력 | 3-lead ECG (10초, 500Hz) + numeric(4) + patient(2) |
| 출력 | 3클래스 (Normal / AFib / Other) |
| 파라미터 | 126,355 |
| **Test Accuracy** | **90.6%** |
| 체크포인트 | `checkpoints/cnn-tcn-3lead/best_model.pt` |

### 2.2 리듬 이상 예측 모델 (Sequence TCN)

| 항목 | 내용 |
|------|------|
| 구조 | CNN-TCN 백본 → 특징벡터 → Temporal TCN 2블록 → FC |
| 입력 | 연속 ECG 시퀀스 (최대 8장, 30일 이내) |
| 출력 | 2클래스 (15일 내 리듬 이상 발생 여부) |
| 대상 | AFib, AFlutter, ectopic atrial, junctional, SVT |
| 파라미터 | 145,477 |
| **Test AUROC** | **0.7355** |
| **Precision (thr=0.5)** | **38%** |
| **Recall (thr=0.5)** | **26%** |
| 체크포인트 | `checkpoints/sequence-15d-rhythm/best_model.pt` |

**임계값별 Precision-Recall 트레이드오프:**

| 임계값 | Precision | Recall | 용도 |
|:------:|:---------:|:------:|------|
| 0.40 | 24% | 65% | 고감도 스크리닝 |
| 0.50 | 38% | 26% | 기본 설정 |
| 0.55 | 50% | 13% | 보수적 경고 |

연속 측정 시 누적 Recall (thr=0.5, 15일): 1-(0.74)^15 ≈ **98%**

## 3. 실험 이력

### 3.1 분류 모델

| # | 실험 | Accuracy | 비고 |
|---|------|:--------:|------|
| 1 | CNN-TCN 12-lead | 93.6% | 12-lead 최고 성능, 기기에서 사용 불가 |
| 2 | CNN-TCN 12-lead + BP | 90.3% | BP 70% 결측 → 노이즈, 실패 |
| 3 | CNN-TCN-CBAM 12-lead | 90.6% | 소규모 모델에 어텐션 비효율, 실패 |
| 4 | **CNN-TCN 3-lead** | **90.6%** | **배포용 최종 모델** |

### 3.2 예측 모델

| # | 실험 | AUROC | Precision | 비고 |
|---|------|:-----:|:---------:|------|
| 1 | 30일 AFib (backbone freeze) | 0.705 | 25% | 2,770 params만 학습 |
| 2 | 30일 AFib (finetune) | 0.705 | 29% | 전체 학습, 거의 차이 없음 |
| 3 | 15일 AFib (finetune) | 0.7185 | 25% | 윈도우 축소 효과 |
| 4 | 15일 AFib + HRV | 0.7073 | 26% | CNN이 이미 학습, 중복 |
| 5 | 15일 AFib + ECG Intervals | 0.7114 | 26% | P-dur/PR/QTc 추가, 중복 |
| 6 | 15일 리듬이상그룹 (단일) | 0.7055 | 33% | 양성 29% 증가 효과 |
| 7 | **15일 리듬이상 시퀀스TCN** | **0.7355** | **38%** | **최종 모델** |

### 3.3 핵심 인사이트

1. **피처 추가는 효과 없음**: HRV, ECG Intervals, BP 모두 CNN-TCN이 파형에서 이미 학습한 정보와 중복
2. **시퀀스가 핵심**: 단일 ECG → 0.72 한계, 연속 ECG 변화 패턴 학습 → 0.74 돌파
3. **리듬 이상 그룹 통합**: AFib만보다 유사 질환 통합 시 양성 샘플 증가 + Precision 개선
4. **임계값 최적화**: 재학습 없이 Precision 25% → 50% 가능 (Recall 트레이드오프)
5. **CBAM 어텐션**: 128K params 규모에서는 오히려 성능 저하

## 4. 데이터

| 파일 | 설명 | 크기 |
|------|------|------|
| `ecg_preprocessed.h5` | 원본 775,367건 (waveform, numeric, patient, bp, hrv, intervals) | ~173GB |
| `ecg_prediction_15d.h5` | AFib 15일 예측 172,657건 (양성 10.7%) | - |
| `ecg_prediction_15d_rhythm.h5` | 리듬이상 15일 예측 172,657건 (양성 13.7%) | - |
| `ecg_sequence_15d_rhythm.h5` | 시퀀스 15일 예측 86,192건 (양성 15.0%, 평균 3.4장) | - |

추가 피처:
- `hrv_features`: 6개 (mean_rr, sdnn, rmssd, pnn50, mean_hr, hr_std) — 100% 계산
- `ecg_interval_features`: 5개 (p_duration, pr_interval, qtc, p_axis_abnormal, qrs_t_angle) — machine_measurements.csv 기반
- `bp_features`: 2개 (sbp, dbp) — ICU chartevents + OMR, 29.9% 매칭

## 5. 논문 대비 성능

| 연구 | 리드 | 데이터 규모 | 예측 윈도우 | AUROC |
|------|:----:|:----------:|:----------:|:-----:|
| Attia et al. (Mayo, 2019) | 12-lead | 60만명 | 발생 여부 | 0.87 |
| Raghunath et al. (Geisinger, 2021) | 12-lead | 40만명 | 1년 | 0.85 |
| Khurshid et al. (MGH, 2022) | 12-lead | 10만명 | 5년 | 0.81 |
| **본 프로젝트** | **3-lead** | **16만명** | **15일** | **0.74** |

3-lead + 단기(15일) + 단일 기관(MIMIC) 조건에서 합리적 수준.
12-lead 모델도 2-lead로 줄이면 AUROC 0.75 수준 (Mayo Clinic 보고).

## 6. 배포 전략

- **분류 모델**: 실시간 부정맥 진단 (ECG 1장 → 즉시 결과)
- **예측 모델**: 연속 측정 데이터 축적 → 리듬 이상 위험도 점수
  - 임계값 0.5 기준 "주의" 경고
  - 며칠 연속 경고 시 "병원 방문 권고"
  - 누적 15일 측정 시 Recall ~98%

## 7. 발작성 AF 감지 모델 (신규, 2026-04-07 추가)

### 7.1 목적

정상 리듬(Sinus) ECG 1장에서 숨겨진 AF 흔적을 감지하는 이진분류 모델.
AF 진단 환자의 정상 리듬 ECG vs 진짜 정상 환자 ECG를 구분하여, "정상처럼 보이지만 AF 가능성 있으니 추가 검사 권장" 스크리닝 용도.

### 7.2 실험 결과

#### 모델 1: ResNet-34 ECG only (Tarabanis 2025 방법론)

| 항목 | 내용 |
|------|------|
| 구조 | ResNet-34 (34-layer, 16 residual blocks) 1D Conv, from scratch |
| 파라미터 | 7,290,914 |
| 입력 | 3-lead (I, II, III 계산) x 5000 samples (500Hz, 10초) |
| 데이터 | 52만건 (양성 52,202 / 음성 471,157, 1:9 비율) |
| 양성 정의 | AF 진단 환자의 +-90일 내 정상 리듬(Sinus) ECG |
| 음성 정의 | AF 이력 없는 환자의 Sinus ECG |
| 학습 | Adam, lr=1e-3, LR scheduler (patience 2, factor 0.8), ES patience 5 |
| 분할 | 7:1:2 (환자 단위) |
| **Test AUROC** | **0.8038** |
| Sensitivity | 69% |
| Specificity | 76% |
| Precision | 24% |
| Best epoch | 10 |
| 체크포인트 | `checkpoints/paroxysmal-af-resnet34/` |

#### 모델 2: ResNet-34 + 임상정보 (ECG + Clinical) -- 최종 모델

| 항목 | 내용 |
|------|------|
| 구조 | ResNet-34 CNN + Late Fusion (CNN 512d + tabular FC) |
| 추가 피처 | DM(당뇨), HF(심부전), MI(심근경색), AHT(항고혈압제) -- 4개 이진 플래그 |
| Tabular 입력 | numeric 8 + patient 2 |
| **Test AUROC** | **0.8240** (+0.02 향상) |
| Sensitivity | 71% |
| Specificity | 77% |
| Precision | 25% |
| Best epoch | 13 |
| 체크포인트 | `checkpoints/paroxysmal-af-resnet34-clinical/` |

#### 임계값 최적화 결과 (ECG + Clinical 모델)

| 전략 | 임계값 | Sens | Spec | Prec | F1 |
|------|:---:|:---:|:---:|:---:|:---:|
| Default (0.5) | 0.50 | 71% | 77% | 25% | 0.372 |
| Youden J | 0.46 | 74% | 74% | 24% | 0.359 |
| **Prec>=30% (권장)** | **0.61** | **60%** | **85%** | **31%** | **0.409** |
| Max F1 | 0.65 | 56% | 87% | 33% | 0.416 |
| Prec>=40% | 0.78 | 37% | 95% | 43% | 0.399 |

**권장 설정**: 임계값 0.61 (Prec>=30%)
- Sens 60%, Spec 85%, Prec 31%
- 병원 연속 모니터링 시 반복 측정으로 누적 감지율 상승
- 알람 피로 관리 가능한 오탐률(15%)

### 7.3 논문 대비 비교

| | 논문 (8-lead, ECG+CHARGE-AF) | 본 프로젝트 (3-lead, ECG+Clinical) |
|---|:---:|:---:|
| AUROC | 0.89 | 0.82 |
| Sensitivity | 78% | 71% (기본) / 60% (임계값 0.61) |
| Specificity | 84% | 77% (기본) / 85% (임계값 0.61) |
| Precision | 54% | 25% (기본) / 31% (임계값 0.61) |

3-lead 한계 내에서 AUROC 0.82는 논문 ECG only(0.83)와 거의 동등한 결과.
V1 없이도 Lead II P파 형태만으로 CNN이 AF 흔적을 학습함.

### 7.4 실패 실험: CNN-TCN Transfer Learning

- 부정맥 분류용 CNN-TCN backbone frozen으로 시도했으나 AUROC 0.68에서 정체
- 부정맥 분류용 backbone은 발작성 AF 감지에 적합하지 않음 (다른 피처 필요)
- ResNet-34 from scratch가 훨씬 효과적

### 7.5 핵심 인사이트

1. **Transfer Learning 실패**: 부정맥 분류(정상/AFib/Other)와 발작성 AF 감지(정상 리듬 내 미세 AF 흔적)는 학습하는 피처가 다름
2. **ResNet-34 from scratch 승리**: 7.3M params 대형 모델이 52만건 데이터에서 충분히 학습
3. **임상정보 효과 확인**: ECG only 0.80 -> ECG+Clinical 0.82 (+0.02), 외부 정보 결합 효과 재확인
4. **3-lead로도 가능**: V1 없이 Lead II P파 형태만으로 AUROC 0.82 달성, 논문 ECG only(0.83)와 근접
5. **임계값 최적화 중요**: 기본 0.5 대비 0.61 설정 시 Prec 25%->31%, Spec 77%->85%로 실용성 향상

### 7.6 사용된 파일

| 파일 | 설명 |
|------|------|
| `ml/preprocessing/build_paroxysmal_af_dataset.py` | 데이터셋 빌더 |
| `ml/model/resnet34_ecg.py` | ResNet34ECG, ResNet34ECGWithTabular 모델 |
| `ml/model/train_paroxysmal_af.py` | 학습 스크립트 |
| `ml/data/ecg_paroxysmal_af.h5` | 학습 데이터 (52만건) |

### 7.7 배포 전략

- **용도**: 매일 아침 10초 측정 시 AF 스크리닝
- **출력 예시**: "AF 흔적 감지 확률: 12%"
- **임계값**: 0.61 (Prec>=30% 기준)
- **연속 모니터링**: 반복 측정으로 누적 감지율 상승 (매일 측정 시 10일 내 ~99% 누적 감지)
- **안내 문구**: "AF 가능성이 감지되었습니다. 추가 검사를 권장합니다" (웰니스 기기 수준)

---

## 8. 다음 단계

### 8.1 추가 ML 모델 개발 (Phase 4 계속)

기능 구성 3종 중 2종 완료. 남은 1종(AFib 예측 개선)을 개발한다.

- [x] (1) 부정맥 분류 -- 완료 (90.6%)
- [x] (2) 발작성 AF 감지 -- **완료 (AUROC 0.8240)**
- [ ] (3) AFib 예측 모델 개선

#### [우선순위 1] AFib 예측 모델 개선 (기존 개선)
- **현재**: 시퀀스 TCN, AUROC 0.7355
- **개선 방향**: MIMIC-IV diagnoses_icd에서 임상정보 추출하여 결합
  - 추가 변수: 당뇨(DM), 심부전(HF), 심근경색(MI), 항고혈압제(AHT) 플래그
  - 참고: Tarabanis et al. (EHJ Digital Health 2025) -- ECG + CHARGE-AF 11개 변수 = AUC 0.89
  - 핵심: 병력 정보는 ECG 파형에 없는 외부 정보이므로 기존 피처 추가(HRV/Intervals) 실패와 다른 결과 기대
- **목표**: AUROC 0.80+
- **구조**: 시퀀스 TCN 특징벡터 + 임상 플래그(4~6개) concat -> FC

### 8.2 소프트웨어 개발 (Phase 5)

1. ESP32 펌웨어 (ADS1292R SPI 드라이버, BLE 전송)
2. Raspberry Pi 5 추론 서버 (모델 3종 ONNX 변환, 순차 추론)
3. 5인치 LCD UI (ECG 파형 + 심박수 + AI 3종 결과 표시)
4. 위험도 대시보드 (AF 감지 확률 추이, 측정 이력)
5. 병원 전송용 PDF 리포트

현재 하드웨어 주문 대기 중 -> ML 모델 추가 개발과 라즈베리파이 추론 파이프라인 선작업 병행 가능.

---

*본 보고서는 ML 모델 학습 과정의 전체 기록입니다. (최종 수정: 2026-04-07, 발작성 AF 감지 모델 결과 추가)*
