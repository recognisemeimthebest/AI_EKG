# AFib 예측 모델 방법론 상세 조사 보고서

> 작성일: 2026-04-06
> 작성자: reference-researcher 에이전트
> 목적: 정상 ECG에서 미래 AFib 발생을 예측하는 기존 연구들의 방법론을 상세히 정리하여, 우리 프로젝트(ESP32+ADS1292R 3-lead, MIMIC-IV 기반)의 모델 개선에 활용

---

## 목차
1. [Attia et al. (Mayo Clinic, 2019)](#1-attia-et-al-mayo-clinic-2019)
2. [Raghunath et al. (Geisinger, 2021)](#2-raghunath-et-al-geisinger-2021)
3. [Khurshid et al. (MGH, 2022)](#3-khurshid-et-al-mgh-2022)
4. [Tarabanis et al. (NYU, 2025)](#4-tarabanis-et-al-nyu-2025)
5. [Suzuki et al. (JAHA, 2024) - Serial ECG](#5-suzuki-et-al-jaha-2024---serial-ecg)
6. [Brant et al. (CIRCEP, 2025) - 다국적 검증](#6-brant-et-al-circep-2025---다국적-검증)
7. [Jabbour et al. (EHJ, 2024) - ECG-AI + PGS](#7-jabbour-et-al-ehj-2024---ecg-ai--polygenic-score)
8. [임상정보 결합 방법론 종합 비교](#8-임상정보-결합-방법론-종합-비교)
9. [프로젝트 적용 시사점](#9-프로젝트-적용-시사점)

---

## 1. Attia et al. (Mayo Clinic, 2019)

**논문**: "An artificial intelligence-enabled ECG algorithm for the identification of patients with atrial fibrillation during sinus rhythm: a retrospective analysis of outcome prediction"
**저널**: The Lancet, 2019;394(10201):861-867
**출처**: https://pubmed.ncbi.nlm.nih.gov/31378392/

### 1.1 과제 정의
- **목표**: 정상동율동(NSR) ECG에서 AF의 심방 구조 변화 흔적을 감지 (AF 이력이 있거나 향후 발생할 환자 식별)
- **양성 클래스**: AF가 확인된 환자의 NSR ECG (AF 이전/이후 모두 포함)
- **음성 클래스**: AF 기록이 없는 환자의 NSR ECG
- **예측 윈도우**: 명시적 시간 제한 없음 (AF 진단 이력 유무로 이진 분류). 후속 분석에서 미래 AF 발생 예측으로도 평가

### 1.2 데이터 구성
| 항목 | 내용 |
|------|------|
| 데이터 출처 | Mayo Clinic ECG 데이터베이스 |
| 수집 기간 | 1993.12.31 ~ 2017.07.21 |
| 총 환자 수 | 180,922명 |
| 총 ECG 수 | 649,931건 (NSR만) |
| 훈련 세트 | 454,789 ECGs / 126,526 환자 |
| 검증 세트 | 64,340 ECGs / 18,116 환자 |
| 테스트 세트 | 130,802 ECGs / 36,280 환자 |
| 분할 비율 | 7:1:2 (환자 기준, 데이터 누수 방지) |
| 포함 기준 | 18세 이상, 디지털 NSR ECG, 앙와위 촬영 |
| AF 유병률 (테스트) | 3,051명 (8.4%) AF 확인 |

### 1.3 모델 구조
- **아키텍처**: Custom CNN (ResNet 계열로 추정)
- **입력**: 5,000 x 12 (10초, 500Hz, 12리드) -> zero-padding -> 5,120 x 12
- **처리 방식**:
  1. 시간축(temporal) 방향으로 다수의 1D 컨볼루션 블록 적용 -> 5,120을 640으로 차원 축소
  2. 리드축(spatial) 방향으로 단일 컨볼루션 레이어로 12개 리드 융합
  3. FC 레이어에서 640개 피처를 받아 최종 분류
- **출력**: SoftMax (이진 분류: AF 유/무)
- **파라미터 수**: 미공개

> **참고**: Attia 그룹은 아키텍처 상세를 공개하지 않았으나, 후속 논문(Attia et al., Circ Arrhythm Electrophysiol, 2019)에서 동일 구조를 나이/성별 예측에도 사용. 5120x12 입력, 640 피처 추출 구조 확인됨.

### 1.4 학습 방법
| 항목 | 내용 |
|------|------|
| Optimizer | 미공개 (Adam 추정) |
| Learning Rate | 미공개 |
| Batch Size | 미공개 |
| Loss Function | 미공개 (Binary CE 추정) |
| 데이터 불균형 처리 | 명시적 기법 없음 |
| Augmentation | 미공개 |
| 모델 선택 | Validation set에서 AUC 기반 임계값 선택 후 Test에 적용 |

### 1.5 성능
| 메트릭 | 값 (95% CI) |
|--------|-------------|
| AUROC | 0.87 (0.86-0.88) |
| Sensitivity | 79.0% (77.5-80.4) |
| Specificity | 79.5% (79.0-79.9) |
| F1 Score | 39.2% (38.1-40.3) |
| Accuracy | 79.4% (79.0-79.9) |

**다수 ECG 활용 시**: 같은 환자의 여러 NSR ECG 중 하나라도 양성이면 AF로 판정 -> AUC 0.87 -> 0.90, F1 39.2% -> 45.4%로 향상

### 1.6 핵심 인사이트
- **AF 이전 ECG에서도** 모델이 양성 판정 -> AF가 발생하기 전 구조적 변화를 포착
- 임상정보 없이 순수 ECG 파형만으로 AUC 0.87 달성
- **다수 ECG 앙상블**이 단일 ECG보다 유의하게 우수 (Serial ECG 접근의 근거)
- F1이 39.2%로 낮음 -> 낮은 양성률에 의한 precision 저하 (임상 스크리닝 목적에서는 허용)

---

## 2. Raghunath et al. (Geisinger, 2021)

**논문**: "Deep Neural Networks Can Predict New-Onset Atrial Fibrillation From the 12-Lead ECG and Help Identify Those at Risk of Atrial Fibrillation-Related Stroke"
**저널**: Circulation, 2021;143(13):1287-1298
**출처**: https://pmc.ncbi.nlm.nih.gov/articles/PMC7996054/

### 2.1 과제 정의
- **목표**: AF 이력 없는 환자에서 1년 내 새로운 AF 발생 예측
- **양성 클래스**: ECG 이후 1년 내 새로운 AF 진단
- **음성 클래스**: 관찰 기간 내 AF 미발생
- **예측 윈도우**: 1년 (주요), 30년 생존분석도 수행
- **특이점**: AF 이력 있는 환자의 NSR ECG는 제외 (순수 "예측" 과제)

### 2.2 데이터 구성
| 항목 | 내용 |
|------|------|
| 데이터 출처 | Geisinger Health System MUSE 데이터베이스 |
| 수집 기간 | 1984.01 ~ 2019.06 |
| 스크리닝 규모 | 2.8M ECGs → 1.6M 사용 |
| 총 환자 수 | ~430,000명 |
| AF 유병률 (테스트) | ~4% (AUPRC baseline) |
| 중앙 추적기간 | 4.1년 (IQR 1.5-8.5) |
| 포함 기준 | 18세 이상, 유의한 아티팩트 없는 12-lead ECG |
| 제외 기준 | 기존 AF 진단, 동시 AF 기록, 수술 후 30일 내 AF, 갑상선항진 관련 AF (1년 내) |

### 2.3 모델 구조
- **아키텍처**: Deep CNN, **3개의 시간적 분기(temporal branch)** 구조
  - Branch 1: 0~5초 (12리드 x 2.5초 + 리드 II/V1/V5 x 2.5초)
  - Branch 2: 5~7.5초
  - Branch 3: 7.5~10초
- **입력**: 15개 신호 트레이스 (12리드 2.5초 + II/V1/V5 10초 리듬 스트립)
- **샘플링**: 500Hz (원래 250Hz인 42%는 선형 보간으로 리샘플링)
- **해상도**: 1 uV
- **Lead I**: Goldberger 방정식으로 2.5~5초 구간 계산
- **변형 모델**:
  - DNN-ECG: ECG만 사용
  - DNN-ECG-AS: ECG + 나이 + 성별
- **아키텍처 상세**: Supplementary Figure II에 도시되었으나 본문에 레이어 상세 미공개

### 2.4 학습 방법
| 항목 | 내용 |
|------|------|
| Optimizer | 미공개 |
| Learning Rate | 미공개 |
| Batch Size | 미공개 |
| Loss Function | 미공개 |
| Early Stopping | 사용 (reference 48에서 인용) |
| 교차검증 | 5-fold cross-validation (main model은 holdout) |
| 임계값 | F2 score 기준 (recall 중시) |

### 2.5 성능
| 모델 | AUROC (95% CI) | AUPRC (95% CI) | Sens | Spec |
|------|----------------|-----------------|------|------|
| DNN-ECG | 0.83 (0.83-0.84) | 0.21 (0.20-0.22) | - | - |
| DNN-ECG-AS | 0.85 (0.84-0.85) | 0.22 (0.21-0.24) | - | - |
| F2 임계점 기준 | - | - | 69% | 81% |

**30년 생존분석**: 모델 양성 vs 음성 HR 7.2 (6.9-7.6)
**NNS**: F2 임계점에서 1년 내 AF 1건 발견에 NNS=9

### 2.6 핵심 인사이트
- **나이+성별만 추가**해도 AUC 0.83 -> 0.85로 향상 (임상 변수의 즉각적 효과)
- 3-branch 시간적 분기 구조가 ECG의 서로 다른 시간 구간을 독립적으로 학습
- AUPRC가 0.21-0.22로 매우 낮음 -> AF 1년 발생률 ~4%의 극심한 불균형 반영
- AF 고위험 환자의 3년 내 뇌졸중도 예측 가능 (62% 감지)

---

## 3. Khurshid et al. (MGH, 2022)

**논문**: "ECG-Based Deep Learning and Clinical Risk Factors to Predict Atrial Fibrillation"
**저널**: Circulation, 2022;145(2):122-133
**출처**: https://pmc.ncbi.nlm.nih.gov/articles/PMC8748400/

### 3.1 과제 정의
- **목표**: 5년 내 incident AF 위험 예측 (생존분석 프레임워크)
- **양성 클래스**: 관찰 기간 중 AF 발생 (ICD 코드 + ECG 리포트 조합, PPV 92%)
- **음성 클래스**: AF 미발생 (중도절단 포함)
- **예측 윈도우**: 5년 (MGH/BWH), 2년 (UK Biobank, 추적기간 제한)
- **핵심 차별점**: CHARGE-AF와의 결합 모델(CH-AI) 최초 검증

### 3.2 데이터 구성
| 항목 | 내용 |
|------|------|
| 데이터 출처 | MGH (내부), BWH + UK Biobank (외부) |
| 훈련 환자 | 45,770명 (MGH, 2년 이상 1차진료 환자) |
| 훈련 ECGs | ~100,954건 |
| 훈련/검증 분할 | 36,081 / 9,689명 |
| 내부 테스트 | 4,166명 (MGH) |
| 외부 테스트 1 | 37,963명 (BWH) |
| 외부 테스트 2 | 41,033명 (UK Biobank) |
| AF 발생률 | 훈련: 2,171건/45,770명 (4.7%), 테스트: 2,424건/83,162명 (2.9%) |
| 포함 기준 | 18-90세, 2회 이상 1차진료, 3년 내 1회 이상 ECG |
| 제외 기준 | 기존 AF 이력 환자 |

### 3.3 모델 구조
- **ECG-AI**: CNN (생존 예측용)
  - 입력: 5,000 x 12 (10초, 500Hz, 12리드)
  - 낮은 샘플링 ECG는 500Hz로 선형 리샘플링
  - 짧은 ECG는 5,000 측정치까지 zero-padding
  - 최종 입력 텐서: 5,000 x 12
  - 아키텍처 상세: Supplemental Figure I (공개되지 않음)
- **Loss Function**: **이산 시간 생존 손실(Discrete-time survival loss)**
  - 시간 빈(time bin) 내 AF 발생의 negative log-likelihood 최적화
  - 중도절단(censoring)을 자연스럽게 처리
- **Multi-task Learning**: AF 예측 외에 **나이 추정, 성별 분류, AF 리듬 감지** 3개 보조 과제 동시 학습
  - Multi-task 접근이 AF 예측 성능을 향상시킨 것으로 보고
- **Dropout**: 사용 (Srivastava 2014 인용, 구체적 비율 미공개)

### 3.4 CHARGE-AF 결합 방법 (CH-AI)
```
방법: Cox 비례위험 모델 프레임워크
1. ECG-AI 확률 출력 → logit 변환 (log-hazard와 선형 관계 확보)
2. Cox 모델 공변량으로 투입:
   - 모델 a: ECG-AI 확률만
   - 모델 b: ECG-AI 확률 + CHARGE-AF 점수 (= CH-AI)
3. CHARGE-AF 개별 구성요소도 별도 분석 (Supplemental Table XIII)
```

**CHARGE-AF 변수 11개**: 나이, 인종, 키, 체중, SBP, DBP, 현재흡연, 항고혈압제, 당뇨, MI 병력, HF 병력

### 3.5 학습 방법
| 항목 | 내용 |
|------|------|
| Optimizer | 미공개 |
| Learning Rate | 미공개 |
| Batch Size | 미공개 |
| Loss | Discrete-time survival (negative log-likelihood) |
| Multi-task | AF 예측 + 나이 + 성별 + AF 리듬 감지 |
| Augmentation | 미공개 |
| 불균형 처리 | Survival loss가 censoring 암묵적 처리 |

### 3.6 성능
| 모델 | MGH (내부) | BWH (외부) | UK Biobank (외부) |
|------|-----------|-----------|------------------|
| ECG-AI | 0.823 | 0.747 | 0.705 |
| CHARGE-AF | 0.802 | 0.752 | 0.732 |
| **CH-AI (결합)** | **0.838** | **0.777** | **0.746** |

**임계값별 분석** (Khurshid 2022 보충자료):
- ECG-AI 상위 5%: HR 3.3-3.8 (5년 AF 발생)
- CH-AI 상위 5%: HR 4.0-4.5

**Calibration**: UK Biobank에서 재보정(recalibration) 후 ICI 7.1x10^-5 (우수)

### 3.7 핵심 인사이트
- **ECG-AI와 CHARGE-AF는 상보적**: 둘의 상관계수 0.41 (MGH) -> 서로 다른 정보 포착
- **Multi-task learning**이 AF 예측 단독 학습보다 성능 향상
- **Discrete-time survival loss**는 중도절단 + 시간까지 고려 -> 이진 분류보다 적합
- CHARGE-AF 개별 구성요소를 ECG-AI와 조합해도 CH-AI와 거의 동일한 판별력 (Supplemental Table XIII)
- 외부 검증에서 성능 하락 (0.823 -> 0.705~0.747) -> 일반화 과제 존재

---

## 4. Tarabanis et al. (NYU, 2025)

**논문**: "Artificial intelligence-enabled sinus electrocardiograms for the detection of paroxysmal atrial fibrillation benchmarked against the CHARGE-AF score"
**저널**: European Heart Journal - Digital Health, 2025;6(6):1134
**출처**: https://pmc.ncbi.nlm.nih.gov/articles/PMC12629645/

> **주의**: 이 논문은 미래 AF "예측"이 아닌 발작성 AF "감지" (+-90일 NSR ECG에서 AF 존재 추정)

### 4.1 과제 정의
- **목표**: 정상동율동 ECG에서 발작성 AF 존재 감지
- **양성 클래스**: AF ECG 전후 +-90일 이내에 기록된 NSR ECG
- **음성 클래스**: AF 기록이 없는 환자의 모든 NSR ECG
- **예측 윈도우**: +-90일 (감지 과제)

### 4.2 데이터 구성
| 항목 | 내용 |
|------|------|
| 데이터 출처 | NYU Langone Health (GE MUSE XML) |
| 수집 기간 | 2012.01 ~ 2022.01 |
| 총 ECG 수 | 157,192건 NSR ECG |
| 총 환자 수 | 76,986명 |
| 분할 비율 | 7:1:2 (train:val:test) |
| 테스트 세트 | 31,693 ECGs / 15,343 환자 / AF 3,064명 (20%) |
| ECG 전처리 | 500Hz -> 250Hz 다운샘플링 |

### 4.3 모델 구조
```
[8x2500 ECG Matrix] --> [34-layer ResNet CNN (16 residual connections)]
                              |
                        [1D representation]
                              |
                    [Concatenation] <-- [CHARGE-AF 11 variables]
                              |
                     [FC Neural Network]
                              |
                        [SoftMax Output]
```

- **백본**: ResNet-34 (34층, 16개 residual connection)
- **ECG 입력**: 8 x 2500 (8 measured leads: I, II, V1-V6, 250Hz, 10초)
- **Fusion 방식**: **Late fusion (concatenation)**
  - CNN이 ECG에서 1D representation 추출
  - 이 representation에 tabular 데이터를 concatenate
  - 결합된 벡터를 FC layer에 통과 -> SoftMax 출력

### 4.4 CHARGE-AF 변수 상세 (11개)
| 번호 | 변수 | 타입 |
|------|------|------|
| 1 | Age (나이) | 연속 |
| 2 | Race (인종) | 범주 |
| 3 | Height (키) | 연속 |
| 4 | Weight (체중) | 연속 |
| 5 | Systolic BP | 연속 |
| 6 | Diastolic BP | 연속 |
| 7 | Current smoking | 이진 |
| 8 | Anti-hypertensive medication | 이진 |
| 9 | Diabetes mellitus | 이진 |
| 10 | MI history | 이진 |
| 11 | HF history | 이진 |

### 4.5 5개 모델 변형 (Ablation)
| 모델 | 입력 | 테스트 AUC | 테스트 AUPRC |
|------|------|-----------|-------------|
| ECG + CHARGE-AF (전체) | ECG + 11변수 | **0.89** | **0.69** |
| ECG + PMH | ECG + DM/MI/HF/항고혈압제 | ~0.87 | ~0.65 |
| ECG + Demographics | ECG + 나이/인종/흡연 | ~0.86 | ~0.62 |
| ECG + Vitals | ECG + 키/체중/BP | ~0.86 | ~0.61 |
| ECG only | ECG만 | 0.83 | 0.54 |
| CHARGE-AF score (tabular) | 11변수만 | 최하위 | - |

### 4.6 학습 방법
| 항목 | 내용 |
|------|------|
| Optimizer | **Adam** |
| LR Scheduler | 2 epochs 개선 없으면 LR x 0.8 |
| Epochs | 100 |
| Loss | **Categorical Cross-Entropy** |
| Early Stopping | Validation loss 5 epochs 미개선 시 중단 |
| Batch Size | 미보고 |
| Class Weighting | 미보고 |
| Augmentation | 미보고 |
| Brier Score | 0.174 |

### 4.7 외부 검증
| 코호트 | N | AUC | AUPRC |
|--------|---|-----|-------|
| NYU 내부 테스트 | 31,693 | 0.89 | 0.69 |
| US Suburban (NYU Long Island) | 5,488 | 0.90 | 0.67 |
| Greece | 306 | 0.85 | 0.78 |

### 4.8 핵심 인사이트
- **PMH(질환 병력)가 가장 큰 기여**: DM, MI, HF, 항고혈압제 4개 변수 추가 시 최대 AUC 향상
- Demographics > Vitals 순서의 기여도
- ECG only(0.83) -> ECG+CHARGE-AF(0.89)로 **+0.06 AUC** 향상 (매우 유의)
- Late fusion (concatenation)이 효과적 -> 구현 단순성 + 성능 동시 확보
- CHARGE-AF features 랜덤 셔플 시에도 성능 크게 저하되지 않음 -> EHR 데이터 품질 이슈에 강건

---

## 5. Suzuki et al. (JAHA, 2024) - Serial ECG

**논문**: "Machine Learning Algorithm to Predict Atrial Fibrillation Using Serial 12-Lead ECGs Based on Left Atrial Remodeling"
**저널**: Journal of the American Heart Association, 2024;13(19):e034154
**출처**: https://pmc.ncbi.nlm.nih.gov/articles/PMC11681470/

### 5.1 과제 정의
- **목표**: Serial 12-lead ECG 쌍에서 좌심방 리모델링 패턴을 감지하여 AF 발생 예측
- **양성 클래스**: AF ECG가 확인된 환자, index AF ECG 이전의 NSR ECG 쌍
- **음성 클래스**: AF 기록 없는 환자의 NSR ECG 쌍
- **예측 윈도우**: 2년 (마지막 NSR ECG부터)
- **핵심 가설**: AF 직전 좌심방 리모델링이 ECG에 반영되며, serial ECG의 변화(delta)가 단일 ECG보다 정확

### 5.2 데이터 구성
| 항목 | 내용 |
|------|------|
| 스크리닝 | 2,162,637 ECGs / 894,356 환자 |
| 사용 데이터 | 415,964 ECGs / 176,090 환자 |
| 개발 데이터 | 67,269 NSR + 11,810 AF 환자 |
| 수집 기간 | 2010.01 ~ 2021.05 |
| 분할 비율 | 8:1:1 (train:val:test, 환자 기준 비중복) |
| 외부 검증 | 1,000 환자 (Samsung Medical Center + Wonju Severance) |
| 포함 기준 | 18세 이상, 표준 12-lead ECG |
| 제외 기준 | 기존 AF 진단, index AF ECG 이전 NSR 없음, NSR 1건만, 불충분 의무기록 |

### 5.3 Serial ECG 정의
- **ECG 쌍**: 동일 환자의 2개 NSR ECG
- **최소 간격 (Blanking Period)**: **3개월** (최적)
  - 1개월: 충분한 ECG 확보 어려움
  - 3개월: 최적 성능 + 충분한 데이터
  - 간격이 길어질수록 성능 하락 -> AF 직전 리모델링이 가장 정보량 높음
- **쌍 생성**: 모든 가능한 조합 방법 (2 ECG x 2 ECG = 4쌍)
- **최적 ECG 간 간격**: 4~14개월 (AUROC 0.968 최고)

### 5.4 모델 구조
- **알고리즘**: **LightGBM** (gradient boosting, 딥러닝 아님)
- **하이퍼파라미터**: Bayesian optimization으로 튜닝
- **피처 선택**: SHAP 분석 기반 상위 15개 -> 최적 20개 피처
- **임계값**: Youden J statistic

### 5.5 피처 상세

**단일 ECG 피처**:
- P-QRS-T 파형 성분: peaks, intervals, segments, durations
- 기술통계: 평균, 최소, 최대, 표준편차
- **P-wave 형태**: skewness, kurtosis
- Beat-to-beat 변화: peaks, intervals, durations의 박동간 변동
- 상관통계: 박동별 Pearson 계수
- F-wave 지수
- HRV 기술통계
- 나이, 성별

**Delta 피처 (Serial ECG)**:
- 쌍 ECG 간 peaks, intervals, durations의 **변화량**
- 좌심방 리모델링의 진행을 포착하기 위한 핵심 피처

### 5.6 성능
| 메트릭 | Single ECG | Serial ECG | p-value |
|--------|-----------|-----------|---------|
| **내부 AUROC** | **0.910** | **0.960** | <0.001 |
| Sensitivity | 0.859 | 0.859 | - |
| Specificity | 0.799 | 0.924 | - |
| Accuracy | 0.846 | 0.894 | - |
| F1 Score | 0.565 | 0.811 | - |
| **외부 AUROC** | **0.812** | **0.880** | <0.001 |
| 외부 Sensitivity | 0.744 | 0.810 | - |
| 외부 Specificity | 0.742 | 0.822 | - |
| 외부 F1 | 0.743 | 0.815 | - |

### 5.7 피처 중요도 (SHAP)
- **나이와 성별이 개별 ECG 파라미터보다 중요**
- 단일 ECG 모델: P-wave duration, amplitude이 상위 10
- Serial ECG 모델: **P-wave duration이 상위 5** ECG 파라미터
- 시간 구간별 분석: 단기/중기/장기 간격에서 피처 중요도 차이 (Figure S2)

### 5.8 핵심 인사이트
- Serial ECG가 단일 ECG 대비 **AUROC +0.05 (내부), +0.068 (외부)** 향상 (매우 유의)
- F1 score 0.565 -> 0.811로 **극적 개선** (specificity 대폭 향상)
- **Blanking period 3개월이 최적** -> AF 직전의 급격한 리모델링 포착
- ECG 간 간격 4-14개월이 최적 -> 너무 짧으면 변화 부족, 너무 길면 노이즈
- LightGBM + hand-crafted features가 end-to-end DL보다 높은 성능 (0.960 vs ~0.85)
- **임상 변수 미사용** (나이/성별만) -> 임상정보 추가 시 추가 향상 여지

---

## 6. Brant et al. (CIRCEP, 2025) - 다국적 검증

**논문**: "Prediction of Atrial Fibrillation From the ECG in the Community Using Deep Learning: A Multinational Study"
**저널**: Circulation: Arrhythmia and Electrophysiology, 2025
**출처**: https://pmc.ncbi.nlm.nih.gov/articles/PMC12569998/

### 6.1 과제 정의
- **목표**: 지역사회 코호트에서 ECG 기반 DL 모델의 5년 AF 예측 + CHARGE-AF 비교/결합
- **양성 클래스**: 5년 내 incident AF
- **예측 윈도우**: 5년

### 6.2 데이터 구성
| 코호트 | N | AF 발생률 |
|--------|---|----------|
| CODE (사전학습) | 631,514 no-AF + 41,851 prevalent + 12,280 incident | - |
| FHS (파인튜닝) | 6,036 (60%) / 4,061 (40% 검증) | 4.6/1000 py |
| UK Biobank | 49,280 | 3.9/1000 py |
| ELSA-Brasil | 12,284 | 1.5/1000 py |

### 6.3 모델 구조
- **아키텍처**: ResNet CNN
- **사전학습**: CODE 데이터셋 (브라질 60만건)에서 3-class 분류 (no-AF / prevalent AF / incident AF)
- **파인튜닝**: FHS 60%에서 LR 0.001 -> 0.0001로 감소
- **입력**: 12-lead, 500Hz, 10초
- **정규화**: Dropout + weight decay
- **교차검증**: 10-fold
- **CHARGE-AF 결합**: 통합 위험점수 (정확한 방법 미상세) -> IDI (Integrated Discrimination Improvement)로 평가

### 6.4 성능
| 코호트 | ECG-AF AUC | CHARGE-AF AUC | **결합 AUC** |
|--------|-----------|---------------|-------------|
| FHS | 0.82 | 0.83 | **0.85** |
| UK Biobank | 0.73 | 0.78 | **0.81** |
| ELSA-Brasil | 0.72 | 0.79 | **0.81** |

ECG-AF와 CHARGE-AF 상관: r=0.41 (FHS), 0.27 (UK), 0.14 (ELSA)

### 6.5 핵심 인사이트
- **ECG-AI 단독은 지역사회 코호트에서 CHARGE-AF보다 낮거나 유사** (병원 데이터로 학습하면 지역사회에서 성능 하락)
- 결합 시 일관되게 향상 (모든 코호트에서 +0.02~0.08)
- **ECG-AF와 CHARGE-AF의 약한 상관** -> 서로 독립적 정보 포착 확인
- ECG-AF는 AF 이외에도 HF, MI, 뇌졸중, 사망 예측에도 유의 -> ECG에 포착되는 것이 AF 특이적이 아닌 전반적 심혈관 위험
- 사전학습(CODE) -> 파인튜닝(FHS)의 **transfer learning** 접근

---

## 7. Jabbour et al. (EHJ, 2024) - ECG-AI + Polygenic Score

**논문**: "Prediction of incident atrial fibrillation using deep learning, clinical models, and polygenic scores"
**저널**: European Heart Journal, 2024;45(46):4920
**출처**: https://academic.oup.com/eurheartj/article/45/46/4920/7740534

### 7.1 과제 정의
- **목표**: 5년 incident AF 예측, ECG-AI vs CHARGE-AF vs PGS 비교
- **예측 윈도우**: 5년

### 7.2 데이터 구성
| 항목 | 내용 |
|------|------|
| 데이터 출처 | Montreal Heart Institute (MHI) |
| 총 ECGs | 669,782건 / 145,323 환자 |
| AF 유병률 | 환자 15.6%, ECG 12.0% |
| 분할 비율 | 70:10:20 (환자 비중복) |
| 평균 나이 | 61 +/- 15세, 58% 남성 |
| 외부 검증 | MIMIC-IV: 109,870 환자, 437,323 ECGs |

### 7.3 모델 구조
- **아키텍처**: **ResNet-50** (random weight initialization)
- **입력**: 12-lead, 250Hz, 10초
- **전처리**: 평균 제거 + 단위 분산 스케일링 (훈련셋 기준), +-10mV 초과 ECG 제외
- **학습**: TensorFlow 2.9.1, 4x A6000 GPU
- **하이퍼파라미터**: Bayesian grid-search로 최적화
- **CHARGE-AF 결합**: **로지스틱 회귀** (ECG-AI 확률 + CHARGE-AF 점수를 공변량으로)
- **PGS 결합**: Khera AF-PGS (650만 SNP) -> 표준화 + logistic 변환 -> 로지스틱 회귀에 추가

### 7.4 성능
| 모델 | MHI 내부 AUC | MIMIC-IV 외부 AUC |
|------|-------------|------------------|
| ECG-AI (ResNet-50) | 0.78 | **0.77** |
| CHARGE-AF | 0.62 | - |
| PGS | 0.59 | - |
| ECG-AI + CHARGE-AF | 0.76-0.77 | - |
| ECG-AI + CHARGE-AF + PGS | 0.76-0.77 | - |

> **주의**: CHARGE-AF/PGS 추가 시 AUC 거의 변화 없으나 모델 적합도(likelihood ratio) 유의하게 개선

### 7.5 핵심 인사이트
- **ResNet-50이 ECG 기반 AF 예측에서 가장 표준적인 아키텍처**로 확립
- ECG-AI가 CHARGE-AF(0.62)와 PGS(0.59)를 크게 능가 (0.78)
- **MIMIC-IV 외부 검증에서 AUC 0.77** -> 우리 데이터와 직접 비교 가능한 벤치마크
- CHARGE-AF/PGS 추가의 AUC 향상은 미미 -> ECG가 이미 임상정보를 암묵적으로 인코딩?
- CHARGE-AF 단독 성능이 0.62로 다른 연구(0.80) 대비 매우 낮음 -> 데이터 특성/구현 차이 가능

---

## 8. 임상정보 결합 방법론 종합 비교

### 8.1 결합 방법론 분류

| 방법 | 연구 | 구현 복잡도 | 장점 | 단점 |
|------|------|-----------|------|------|
| **Late Fusion (Concatenation)** | Tarabanis 2025 | 낮음 | 단순, 효과적 (AUC +0.06) | 모달리티 간 interaction 제한 |
| **Cox PH Model** | Khurshid 2022 | 중간 | 생존분석 자연스럽게 처리, 해석 가능 | 비선형 interaction 불가 |
| **Logistic Regression** | Jabbour 2024, Brant 2025 | 낮음 | 해석 가능, 사후 결합 가능 | 최적 결합 아닐 수 있음 |
| **입력 추가 (Age/Sex)** | Raghunath 2021 | 매우 낮음 | 즉시 적용 | 제한된 임상정보만 가능 |
| **Hand-crafted + ML** | Suzuki 2024 | 중간 | 피처 해석 가능, 높은 성능 | 피처 엔지니어링 노동 |

### 8.2 임상 변수별 기여도 종합

| 변수 | 기여도 근거 | 핵심 출처 |
|------|-----------|----------|
| **PMH (DM, MI, HF)** | 가장 큰 AUC 향상 | Tarabanis 2025 |
| **나이** | 모든 연구에서 최상위 피처 | Suzuki 2024, Raghunath 2021 |
| **성별** | 나이 다음으로 중요 | Suzuki 2024 |
| 항고혈압제 | PMH 그룹으로 효과적 | Tarabanis 2025 |
| 인종 | Demographics 그룹에서 기여 | Tarabanis 2025 |
| 키/체중/BP | Vitals 그룹에서 최소 기여 | Tarabanis 2025 |
| P-wave duration | ECG 피처 중 상위 | Suzuki 2024 |
| P-wave amplitude | ECG 피처 중 상위 | Suzuki 2024 |
| PGS (유전) | AUC 추가 향상 미미 | Jabbour 2024 |

### 8.3 예측 윈도우별 성능 비교

| 연구 | 윈도우 | 내부 AUC | 외부 AUC | 임상정보 |
|------|--------|---------|---------|---------|
| Attia 2019 | 이력 기반 | 0.87 | - | 없음 |
| Raghunath 2021 | 1년 | 0.85 | - | 나이+성별 |
| Khurshid 2022 | 5년 | 0.838 (CH-AI) | 0.746-0.777 | CHARGE-AF |
| Tarabanis 2025 | +-90일 (감지) | 0.89 | 0.85-0.90 | CHARGE-AF 11개 |
| Suzuki 2024 | 2년 (serial) | 0.960 | 0.880 | 나이+성별 |
| Brant 2025 | 5년 | 0.85 (결합) | 0.81 | CHARGE-AF |
| Jabbour 2024 | 5년 | 0.78 | 0.77 (MIMIC) | CHARGE-AF+PGS |
| **우리 모델** | **15일** | **0.7355** | - | **없음** |

---

## 9. 프로젝트 적용 시사점

### 9.1 즉시 적용 가능한 개선 사항

#### (1) 임상정보 결합 - Late Fusion (Tarabanis 방식)
```
현재: CNN/TCN → ECG only → 예측
개선: CNN/TCN → ECG 피처 + [임상변수 벡터] → FC → 예측

구현:
1. CNN backbone의 마지막 FC 이전 representation 추출
2. 임상 변수 벡터 (나이, 성별, PMH, BP 등) concatenate
3. FC layer(들) 통과 → sigmoid 출력
```

**우선순위 변수 (MIMIC-IV에서 추출 가능)**:
1. **나이, 성별** (즉시, patients 테이블)
2. **PMH 3개**: DM, MI, HF (diagnoses_icd에서 ICD 코드)
3. **항고혈압제** (prescriptions/emar)
4. **SBP/DBP** (chartevents/OMR)
5. **BMI** (키/체중 chartevents/OMR)

#### (2) Loss Function 개선
- 현재: Binary CE
- 개선 옵션:
  - **Discrete-time survival loss** (Khurshid 방식) -> 시간까지 고려, censoring 처리
  - **Focal Loss** (class imbalance 대응)

#### (3) Multi-task Learning (Khurshid 방식)
- AF 예측 + 나이 추정 + 성별 분류 + 리듬 분류 동시 학습
- 공유 representation이 더 풍부한 ECG 피처 학습

#### (4) Serial ECG Delta Features (Suzuki 방식)
- 현재 Sequence TCN이 이미 serial ECG 사용
- 추가: 명시적 delta features (P-wave duration 변화, PR interval 변화 등) 계산 후 입력에 추가
- **Blanking period 3개월 적용** 검토

### 9.2 아키텍처 권장사항

| 구성요소 | 현재 | 권장 | 근거 |
|---------|------|------|------|
| ECG 백본 | CNN-TCN | **ResNet-34/50** | Tarabanis/Jabbour 표준 |
| 임상정보 결합 | 없음 | **Late Fusion (concatenation)** | Tarabanis: 단순+효과적 |
| 시퀀스 처리 | Temporal TCN | Temporal TCN + **delta features** | Suzuki: 명시적 변화량 |
| Loss | BCE | **Discrete-time survival** OR **Focal Loss** | Khurshid: 시간고려 |
| 보조 과제 | 없음 | **나이/성별 multi-task** | Khurshid: 성능 향상 확인 |

### 9.3 기대 성능 향상

현재 AUROC 0.7355 (15일, ECG only) 기준:
- 임상정보 Late Fusion: +0.03~0.06 (Tarabanis 근거)
- Multi-task learning: +0.01~0.02 (Khurshid 근거)
- Delta features: +0.02~0.05 (Suzuki 근거)
- Loss 개선: +0.01~0.02
- **종합 기대**: AUROC 0.78~0.83 (보수적 추정)

### 9.4 주의사항
1. **과제 정의 차이**: Tarabanis(감지)와 우리(예측)는 근본적으로 다른 과제. AUC 직접 비교 주의
2. **리드 수 차이**: 대부분 연구가 12-lead. 우리는 3-lead -> 성능 하한이 있을 수 있음
3. **데이터 규모 차이**: Attia 65만, Raghunath 160만 vs 우리 ~80만. 규모는 유사하나 3-lead 한계
4. **외부 검증 필수**: 모든 논문에서 외부 검증 시 5~10%p AUC 하락. 내부 성능만으로 판단 금지

---

## 참고문헌

1. Attia ZI, et al. An artificial intelligence-enabled ECG algorithm for the identification of patients with atrial fibrillation during sinus rhythm. *Lancet*. 2019;394:861-867. [PubMed](https://pubmed.ncbi.nlm.nih.gov/31378392/)
2. Raghunath S, et al. Deep Neural Networks Can Predict New-Onset Atrial Fibrillation From the 12-Lead ECG. *Circulation*. 2021;143:1287-1298. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7996054/)
3. Khurshid S, et al. ECG-Based Deep Learning and Clinical Risk Factors to Predict Atrial Fibrillation. *Circulation*. 2022;145:122-133. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8748400/)
4. Tarabanis C, et al. AI-enabled sinus ECGs for detection of paroxysmal AF benchmarked against CHARGE-AF. *EHJ-Digital Health*. 2025;6(6):1134. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12629645/)
5. Suzuki et al. ML Algorithm to Predict AF Using Serial 12-Lead ECGs Based on LA Remodeling. *JAHA*. 2024;13(19):e034154. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11681470/)
6. Brant et al. Prediction of AF From the ECG in the Community Using Deep Learning: A Multinational Study. *CIRCEP*. 2025. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12569998/)
7. Jabbour et al. Prediction of incident AF using deep learning, clinical models, and polygenic scores. *EHJ*. 2024;45(46):4920. [Oxford Academic](https://academic.oup.com/eurheartj/article/45/46/4920/7740534)
