# 발작성 AF(Paroxysmal AF) 감지: 주요 연구 방법론 상세 분석

> 작성일: 2026-04-06
> 작성: reference-researcher 에이전트
> 목적: 정상동율동(NSR) ECG에서 발작성 AF를 감지하는 AI 모델의 핵심 논문 방법론을 구현 수준으로 정리

---

## 1. Attia et al. (Mayo Clinic, 2019) — 최초의 랜드마크 연구

**논문**: "An artificial intelligence-enabled ECG algorithm for the identification of patients with atrial fibrillation during sinus rhythm: a retrospective analysis of outcome prediction"
**저널**: The Lancet, 2019;394(10201):861-867
**출처**: [PubMed](https://pubmed.ncbi.nlm.nih.gov/31378392/) | [Lancet](https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(19)31721-0/abstract)

### 1.1 데이터 구성

| 항목 | 상세 |
|------|------|
| **데이터 출처** | Mayo Clinic ECG Laboratory |
| **수집 기간** | 1993.12.31 ~ 2017.07.21 (약 24년) |
| **총 환자 수** | 180,922명 |
| **총 ECG 수** | 649,931건 (정상동율동 ECG만) |
| **양성(AF) 정의** | 해당 ECG 전후로 AF가 기록된 적 있는 환자의 NSR ECG |
| **음성(Non-AF) 정의** | AF 기록이 전혀 없는 환자의 NSR ECG |
| **포함 기준** | 18세 이상, 디지털 12-lead ECG, 앙와위(supine), NSR |
| **리듬 라벨 검증** | 심장전문의 감독 하 훈련된 인력이 검증 |

**데이터 분할**:

| 세트 | ECG 수 | 환자 수 | 비율 |
|------|--------|---------|------|
| Training | 454,789 | 126,526 | 70% |
| Validation | 64,340 | 18,116 | 10% |
| Test | 130,802 | 36,280 | 20% |

- 분할 단위: **환자 레벨** (동일 환자의 ECG가 다른 세트에 섞이지 않음)

### 1.2 모델 구조

| 항목 | 상세 |
|------|------|
| **아키텍처** | Custom CNN (ResNet 계열) |
| **입력 형태** | 12-lead x 10초 x 500Hz = **5,000 x 12 행렬** |
| **핵심 구조** | Long axis 방향 다층 Conv → 형태적/시간적 특징 추출 → Short axis 방향 단일 Conv → 12리드 퓨전 |
| **중간 표현** | 5,000 x 12 입력 → **2,048 x 3 특징 행렬**로 압축 |
| **출력** | Binary classification (AF 있음/없음) |

**아키텍처 상세**:
- 논문 본문에서는 구체적 레이어 수, 필터 크기 등을 공개하지 않음
- Supplementary에 "network architecture diagram" 포함되었으나 공개 접근 제한
- 후속 연구(Christopoulos 2023, Gruwez 2023)에서 동일 모델 재사용/검증
- Mayo Clinic의 다른 AI-ECG 연구(좌심실 기능저하 감지 등)에서도 동일 아키텍처 사용

### 1.3 학습 방법

| 항목 | 상세 |
|------|------|
| **데이터 분할** | 7:1:2 (Train:Val:Test), 환자 레벨 |
| **Optimizer** | 미공개 (supplementary 참조) |
| **Learning Rate** | 미공개 |
| **Batch Size** | 미공개 |
| **Loss Function** | Binary classification loss (추정: BCE) |
| **Class Weighting** | 미공개 |
| **Augmentation** | 미공개 |

### 1.4 성능 (Test Set)

| 조건 | AUC | Sensitivity | Specificity | F1 | Accuracy |
|------|-----|-------------|-------------|-------|----------|
| **단일 ECG** | **0.87** (0.86-0.88) | 79.0% | 79.5% | 39.2% | 79.4% |
| **첫 달 내 모든 ECG** | **0.90** (0.90-0.91) | 82.3% | 83.4% | 45.4% | 83.3% |

- 복수 ECG 활용 시 성능이 유의하게 향상됨 (AUC +0.03)

### 1.5 핵심 인사이트

1. **P파 말단부(terminal P-wave)가 핵심**: Grad-CAM/Saliency 분석에서 P파의 말단부가 모델 판단에 가장 큰 영향
2. **좌심방 구조적 변화 반영**: AF 환자는 NSR 중에도 좌심방 확대/리모델링의 ECG 흔적을 남김
3. **임상적으로 정상으로 읽힌 ECG에서도 감지 가능**: 사람 눈에 완전히 정상인 ECG에서 AI가 AF 흔적 발견
4. **복수 ECG의 이점**: Serial ECG 접근이 단일 ECG보다 우월 (AUC 0.87 → 0.90)

### 1.6 임상 검증 (후속)

**BEAGLE Trial** (Noseworthy et al., Lancet 2022):
- 전향적 비무작위 중재 시험, 1,003명 등록
- AI가 고위험으로 분류한 군: 30일 연속 모니터링에서 **AF 발견율 7.6%**
- AI가 저위험으로 분류한 군: AF 발견율 **1.6%**
- OR = **4.98** (95% CI 2.22-11.75, p=0.0002)
- 평균 모니터링 기간: 22.3일

---

## 2. Raghunath et al. (Geisinger, 2021) — 대규모 예측 연구

**논문**: "Deep Neural Networks Can Predict New-Onset Atrial Fibrillation From the 12-Lead ECG and Help Identify Those at Risk of Atrial Fibrillation-Related Stroke"
**저널**: Circulation, 2021;143(13):1287-1298
**출처**: [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7996054/) | [PubMed](https://pubmed.ncbi.nlm.nih.gov/33588584/)

### 2.1 데이터 구성

| 항목 | 상세 |
|------|------|
| **데이터 출처** | Geisinger Health System, MUSE 데이터베이스 |
| **수집 기간** | 1984.01 ~ 2019.06 (약 35년) |
| **총 ECG 수** | 약 1,600,000건 |
| **총 환자 수** | 약 431,000명 |
| **포함 기준** | 18세 이상, 유의한 아티팩트 없는 ECG |
| **제외 기준** | 기존 AF/심방조동 진단, 심장수술 후 30일 이내, 갑상선기능항진 진단 후 1년 이내 |
| **양성(AF) 정의** | Baseline ECG 이후 최소 1일~최대 1년 이내 새로운 AF 발생 |
| **AF 확인 방법** | 12-lead ECG의 임상 AF 판독 OR 입원/외래 2회 이상 AF 진단코드 OR 문제 목록(problem list) 등재 |
| **심방조동 처리** | AF와 동일 클래스로 그룹화 (임상적 결과 유사성) |

**데이터 분할**:

| 모델 | 방법 | 상세 |
|------|------|------|
| Proof-of-concept (M0) | Holdout | 80% train / 20% test, 환자 레벨 분리 |
| Cross-validation (M1-M5) | 5-fold CV | 각 fold에서 교차 검증 |
| Deployment | 시간 기반 | 1984-2009 train / 2010-2014 test |

### 2.2 모델 구조

| 항목 | 상세 |
|------|------|
| **아키텍처** | Deep CNN, **3-Branch 시간 분할 설계** |
| **입력 형태** | 12-lead ECG, 500Hz, 10초 (250Hz 수집분은 리니어 보간으로 500Hz 리샘플링) |
| **해상도** | 1 uV |

**3-Branch 구조** (핵심 독창성):

| Branch | 시간 구간 | 사용 리드 |
|--------|-----------|-----------|
| Branch 1 | 0-5초 | I, II, V1, V5 (4 leads) |
| Branch 2 | 5-7.5초 | V1, V2, V3, II, V5 (5 leads) |
| Branch 3 | 7.5-10초 | II, V1, V4, V5, V6 (6 leads) |

- 총 **15개 신호 트레이스** 입력
- 각 Branch의 출력을 결합하여 최종 예측
- 상세 레이어/필터 구성은 Data Supplement Figure II에 있으나 본문에 미기재

**전처리**:
- 250Hz → 500Hz 리니어 보간 리샘플링 (42% 해당)
- Lead I 누락 시: Goldberger 방정식으로 aVR, II에서 계산
- 아티팩트 있는 ECG 제외 (최종 판독 기준)

### 2.3 학습 방법

| 항목 | 상세 |
|------|------|
| **Validation** | 훈련 데이터의 20%를 내부 검증으로 분리, AUROC 추적하여 과적합 방지 |
| **Optimizer** | 미공개 |
| **Learning Rate** | 미공개 |
| **Batch Size** | 미공개 |
| **Epochs** | 미공개 |
| **Loss Function** | 미공개 |
| **Class Weighting** | 미공개 |

### 2.4 성능

**Proof-of-concept (M0, 1년 이내 AF 예측)**:

| Metric | DNN-ECG-AS |
|--------|------------|
| **AUROC** | **0.85** (0.84-0.85) |
| AUPRC | 0.22 (0.21-0.24) |

**Deployment 시나리오 (시간 기반 분할)**:

| Metric | 값 |
|--------|-----|
| AUROC | 0.83 |
| AUPRC | 0.17 |
| Sensitivity | 69% |
| Specificity | 81% |
| NNS (AF 발견) | 9 |
| NNS (AF-관련 뇌졸중 3년 내) | 162 |

**시간대별 성능**:

| AF 발생 시점 | AUROC |
|-------------|-------|
| 1-31일 후 (잠복 AF 감지) | **0.87** |
| 31-365일 후 (미래 AF 예측) | 0.84 |

**장기 추적 (30년 생존분석)**:
- 고위험 vs 저위험 Hazard Ratio = **7.2** (95% CI 6.9-7.6)

### 2.5 핵심 인사이트

1. **이중 기능**: 모델은 (1) 잠복 발작성 AF 감지와 (2) 미래 AF 발생 예측을 동시에 수행
2. **정상 판독 ECG에서도 유효**: 임상적으로 정상으로 판독된 ECG 하위 그룹에서도 높은 성능 유지
3. **심방 근병증(atrial myopathy)**: ECG에 반영된 심방 구조적 변화를 DNN이 포착하는 것으로 추정
4. **NNS 9**: AF 1건 발견에 9명만 스크리닝하면 됨 (효율적)
5. **70세 미만에서도 유효**: 1년 내 새 AF의 38%가 70세 미만에서 발생 → 모든 연령대에 적용 가능

---

## 3. Tarabanis et al. (NYU, 2025) — ECG + 임상정보 결합

**논문**: "Artificial intelligence-enabled sinus electrocardiograms for the detection of paroxysmal atrial fibrillation benchmarked against the CHARGE-AF score"
**저널**: European Heart Journal - Digital Health, 2025;6(6):1134
**출처**: [EHJ-DH](https://academic.oup.com/ehjdh/article/6/6/1134/8239531)

### 3.1 데이터 구성

| 항목 | 상세 |
|------|------|
| **데이터 출처** | NYU Langone Health (GE MUSE XML) |
| **수집 기간** | 2012.01 ~ 2022.01 (10년) |
| **총 ECG 수** | 157,192건 (NSR ECG만) |
| **총 환자 수** | 76,986명 |
| **양성(AF) 정의** | MUSE 시스템에서 AF 라벨이 있는 ECG가 1건 이상 존재하는 환자 |
| **양성 ECG 선택** | AF ECG 전후 **+-90일 이내** 기록된 NSR ECG |
| **시간 윈도우** | 180일 이내 복수 AF ECG 발생 시 → 가장 이른(index) AF ECG 기준 |
| **음성(Non-AF) 정의** | AF 기록이 전혀 없는 환자의 모든 NSR ECG |

**데이터 분할**:

| 세트 | 비율 | ECG 수 | 환자 수 | AF 비율 |
|------|------|--------|---------|---------|
| Training | 70% | ~109,000 | ~53,900 | - |
| Validation | 10% | ~15,700 | ~7,700 | - |
| Test | 20% | 31,693 | 15,343 | 20% (3,064명) |

- 분할 단위: **환자 레벨**

### 3.2 모델 구조

| 항목 | 상세 |
|------|------|
| **백본 CNN** | **34-layer ResNet, 16 residual connections** |
| **ECG 입력** | **8 measured leads (I, II, V1-V6) x 10초 x 250Hz = 8 x 2,500 행렬** |
| **리드 선택** | 8개 실측 리드만 사용 (augmented leads aVR, aVL, aVF, III 제외) |
| **ECG 전처리** | 500Hz → 250Hz 다운샘플링 |
| **CNN 출력** | 1D 압축 표현(concise 1D representation) |
| **퓨전 방식** | CNN 1D 출력 + Tabular 데이터 **concatenation** → FC layer → Softmax |
| **출력** | Binary classification (AF/Non-AF) |

**5가지 모델 변형**:

| 모델 | ECG | Vitals | Demographics | PMH | CHARGE-AF |
|------|-----|--------|-------------|-----|-----------|
| ECG only | O | - | - | - | - |
| ECG + Vitals | O | 키, 체중, SBP, DBP | - | - | - |
| ECG + Demographics | O | - | 나이, 인종, 흡연 | - | - |
| ECG + PMH | O | - | - | DM, MI, HF, 항고혈압제 | - |
| ECG + CHARGE-AF | O | O | O | O | 전체 11개 |

### 3.3 학습 방법

| 항목 | 상세 |
|------|------|
| **Optimizer** | **Adam** |
| **Epochs** | **100** |
| **Loss Function** | **Categorical cross-entropy** |
| **LR Scheduler** | Validation loss 2 epoch 동안 미개선 시 LR x 0.8 |
| **Early Stopping** | Validation loss 5 epoch 동안 미개선 시 중단 |
| **모델 선택 기준** | Validation set에서 **평균 AUPRC 최고** 시점 |
| **Batch Size** | 미공개 |
| **Class Weighting** | 미공개 (불균형 데이터 인정하나 처리 방법 명시 안 함) |
| **Augmentation** | 미공개 |

### 3.4 성능

**내부 테스트셋 (NYU Langone)**:

| 모델 | AUC | AUPRC | Sens | Spec | PPV | NPV | F1 | Accuracy |
|------|-----|-------|------|------|-----|-----|----|----------|
| **ECG + CHARGE-AF** | **0.89** (0.88-0.89) | **0.69** (0.67-0.70) | 0.78 | 0.84 | 0.54 | 0.94 | 0.64 | 0.82 |
| ECG only | 0.83 (0.83-0.84) | 0.54 | 0.68 | 0.81 | 0.47 | 0.91 | 0.55 | - |
| CHARGE-AF tabular only | 최하위 | - | - | - | - | - | - | - |

- **Brier Score**: 0.174 (양호한 확률적 보정)

**외부 검증**:

| 코호트 | AUC | AUPRC | Sens | Spec | PPV | NPV | F1 |
|--------|-----|-------|------|------|-----|-----|----|
| US (NYU Long Island, n=5,488) | **0.90** (0.89-0.91) | 0.67 | 0.75 | 0.88 | 0.48 | 0.96 | 0.58 |
| Greece (n=306) | **0.85** (0.81-0.88) | 0.78 | 0.85 | 0.73 | 0.66 | 0.88 | 0.74 |

**Ablation 결과 (기여도 순서)**:

```
PMH (DM, MI, HF, 항고혈압제) >>> Demographics (나이, 인종, 흡연) > Vitals (키, 체중, BP)
```

- PMH 추가 시 AUC, AUPRC, Specificity, PPV, F1 **모두** 가장 큰 향상
- Vitals 추가는 상대적으로 미미한 개선

### 3.5 핵심 인사이트

1. **ECG 단독 vs 결합**: ECG만으로 AUC 0.83, 임상정보 결합 시 0.89 (+ 0.06)
2. **PMH가 최대 기여자**: DM, MI, HF, 항고혈압제 4개 변수가 가장 큰 성능 향상
3. **CHARGE-AF 단독은 부족**: 전통적 임상 점수만으로는 AI-ECG에 못 미침
4. **EHR 미기록 리스크 포착**: ECG가 EHR에 정확히 기록되지 않은 위험인자 또는 미지의 위험 특징을 감지
5. **8 measured leads 충분**: augmented leads 없이도 높은 성능 달성
6. **+-90일 윈도우**: 이 시간 윈도우가 발작성 AF 감지 과제 정의의 표준이 됨

---

## 4. Khurshid et al. (MGH/Broad Institute, 2022) — ECG-AI + CHARGE-AF 통합

**논문**: "ECG-Based Deep Learning and Clinical Risk Factors to Predict Atrial Fibrillation"
**저널**: Circulation, 2022;145(2):122-133
**출처**: [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8748400/) | [PubMed](https://pubmed.ncbi.nlm.nih.gov/34743566/)

### 4.1 데이터 구성

| 항목 | 상세 |
|------|------|
| **훈련 데이터** | MGH 1차진료 환자 36,081명, 100,954 ECG |
| **내부 검증** | 9,689명 |
| **테스트 세트** | MGH 4,166명 + BWH 37,963명 + UK Biobank 41,033명 = **총 83,162명** |
| **양성(AF) 정의** | 5년 이내 incident AF (Modified AF Algorithm, PPV 92%) |
| **UK Biobank** | 2년 이내 incident AF (추적 기간 제한: 중앙값 2.8년) |
| **AF 확인** | 진단코드 + 시술코드 + ECG 보고서 조합 알고리즘 |

### 4.2 모델 구조

| 항목 | 상세 |
|------|------|
| **입력** | 12-lead x 10초 x 500Hz = **5,000 x 12** (저샘플링율은 리니어 리샘플링, 짧은 기록은 zero-padding) |
| **아키텍처** | CNN (상세 미공개, Supplement Figure I) |
| **Loss Function** | 시간-이벤트 손실 (time-to-event): 이산 시간 빈(discrete time bin) + 중도절단(censoring) 고려 |
| **Multi-task** | AF 예측 + 나이 추정 + 성별 분류 + AF 현재 존재 감지 (4개 보조 과제) |
| **CHARGE-AF 통합** | Cox proportional hazards: ECG-AI 출력(logit 변환) + CHARGE-AF 점수 → CH-AI |

### 4.3 성능

**ECG-AI 단독 (5년 AF 예측)**:

| 코호트 | AUC | Average Precision |
|--------|-----|-------------------|
| MGH | 0.823 (0.790-0.856) | 0.27 |
| BWH | 0.747 (0.736-0.759) | 0.19 |
| UK Biobank | 0.705 (0.673-0.737) | 0.06 |

**CHARGE-AF 단독**:

| 코호트 | AUC |
|--------|-----|
| MGH | 0.802 |
| BWH | 0.752 |
| UK Biobank | 0.732 |

**CH-AI (통합)**:

| 코호트 | AUC | Average Precision |
|--------|-----|-------------------|
| MGH | **0.838** | 0.30 |
| BWH | **0.777** | 0.21 |
| UK Biobank | **0.746** | 0.06 |

**Hazard Ratio (1-SD 증가당)**:

| 모델 | MGH | BWH |
|------|-----|-----|
| ECG-AI | 2.45 | 2.05 |
| CHARGE-AF | 3.36 | 2.78 |
| CH-AI | 3.74 | 2.76 |

### 4.4 Saliency Map 분석

- **P파 및 주변 영역**이 AF 위험 예측에 가장 큰 영향
- 고위험 환자의 중앙 파형: **P파 기간 연장, QRS 약간 넓어짐, ST 분절 평탄화**
- ECG-AI와 CHARGE-AF 상관계수: r=0.61 (MGH), 0.66 (BWH), 0.41 (UK Biobank)
- 중등도 상관 → ECG-AI가 CHARGE-AF와 다른 독립적 정보를 추출

### 4.5 핵심 인사이트

1. **Multi-task learning**: 나이/성별/현재AF 보조 과제가 표현 학습 강화
2. **Time-to-event 손실**: 단순 이진 분류가 아닌 시간 빈 기반 생존 분석 접근
3. **ECG-AI vs CHARGE-AF**: 유사한 수준이지만 **중등도 상관** → 결합 시 시너지
4. **보정(calibration) 주의**: UK Biobank에서 위험 과대추정 → **재보정(recalibration) 필수**
5. **데이터 출처 차이에 민감**: 훈련 세트와 유사한 인구일수록 성능 우수

---

## 5. 기타 주요 연구

### 5.1 Christopoulos et al. (Mayo Clinic, 2023) — 대규모 외부 검증

**논문**: "Detecting Paroxysmal Atrial Fibrillation From an Electrocardiogram in Sinus Rhythm: External Validation of the AI Approach"
**저널**: JACC: Clinical Electrophysiology, 2023
**출처**: [JACC](https://www.jacc.org/doi/10.1016/j.jacep.2023.04.008)

| 항목 | 상세 |
|------|------|
| **데이터** | 494,042 ECG, 142,310 환자 |
| **모델** | Attia 2019와 동일 아키텍처 |
| **AUC** | 0.87 (0.86-0.87) |
| **Accuracy** | 78.1% (77.6-78.5%) |
| **AUPRC** | 0.48 (0.46-0.50) |

**위험군별 성능**:

| AF 유병률 | AUPRC |
|-----------|-------|
| 저위험 (3%) | 0.21 |
| 고위험 (30%) | **0.76** |

**AI-ECG 확률 시간 추이** (Christopoulos 2022):

| 시점 | 평균 AI-ECG 확률 |
|------|------------------|
| AF 2-5년 전 | 19.8% |
| AF 1-2년 전 | 23.6% |
| AF 0-3개월 전 | **34.0%** |
| AF 0-3개월 후 | **40.9%** |
| AF 1-2년 후 | 35.2% |
| AF 2-5년 후 | 42.2% |

- AF 접근 시 AI 확률이 점진적으로 상승 → **심방 리모델링의 진행성 포착**

### 5.2 Gruwez et al. (Belgium, 2023) — 독립 재현 연구

**논문**: "Detecting Paroxysmal Atrial Fibrillation From an Electrocardiogram in Sinus Rhythm: External Validation of the AI Approach"
**저널**: JACC: Clinical Electrophysiology, 2023

| 항목 | 상세 |
|------|------|
| **데이터** | ZOL (벨기에 Genk) + ZMK (벨기에 Maaseik) |
| **방법** | Attia 2019 방법론을 **독립적으로 재구축** |
| **AUC** | **0.87** |
| **성별 차이** | 여성 0.90 vs 남성 0.84 |

### 5.3 Brant et al. (다국적, 2025) — 커뮤니티 기반 검증

**논문**: "Prediction of Atrial Fibrillation From the ECG in the Community Using Deep Learning: A Multinational Study"
**저널**: Circulation: Arrhythmia and Electrophysiology, 2025
**출처**: [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12569998/)

| 항목 | 상세 |
|------|------|
| **아키텍처** | ResNet-based CNN |
| **사전훈련** | CODE 데이터셋 (631,514 non-AF + 41,851 prevalent AF + 12,280 incident AF ECGs) |
| **미세조정** | FHS 60% (6,036명, 16,876 ECGs), lr=0.0001 |
| **예측 대상** | 5년 이내 incident AF |

**성능 (ECG-AF 단독)**:

| 코호트 | AUC | 특징 |
|--------|-----|------|
| FHS | **0.82** (0.80-0.84) | 미국 백인 중심 |
| UK Biobank | 0.73 (0.71-0.75) | 유럽 다국적 |
| ELSA-Brasil | 0.72 (0.66-0.78) | 브라질 다인종 |

**CHARGE-AF 결합 시**:

| 코호트 | ECG-AF | CHARGE-AF | Combined |
|--------|--------|-----------|----------|
| FHS | 0.82 | 0.83 | **0.85** |
| UK Biobank | 0.73 | 0.78 | **0.81** |
| ELSA-Brasil | 0.72 | 0.79 | **0.81** |

- **사전훈련(pre-training) + 미세조정(fine-tuning)** 전략 사용
- ECG-AF와 CHARGE-AF의 상관: r = 0.41 (FHS), 0.27 (UK Biobank), 0.14 (ELSA-Brasil)
- 낮은 상관 → 독립적 정보원 → **결합 시 일관된 성능 향상**

---

## 6. 종합 비교 분석

### 6.1 과제 정의 비교

| 연구 | 과제 유형 | 양성 정의 | 시간 윈도우 | 핵심 차이 |
|------|-----------|-----------|------------|-----------|
| **Attia 2019** | 잠복 AF 감지 | AF 이력 있는 환자의 NSR ECG | 전체 이력 | 최초 연구, 시간 제한 없음 |
| **Raghunath 2021** | 신규 AF 예측 | Baseline 후 AF 발생 | **1년** | 예측 과제, 발생 시점 명시 |
| **Tarabanis 2025** | 잠복 AF 감지 | AF ECG 전후의 NSR ECG | **+-90일** | 엄격한 시간 윈도우 |
| **Khurshid 2022** | AF 발생 예측 | Incident AF | **5년** | Time-to-event 접근 |
| **Brant 2025** | AF 발생 예측 | Incident AF | **5년** | 다국적 검증 |

### 6.2 모델 아키텍처 비교

| 연구 | 아키텍처 | 리드 수 | 샘플링 | 입력 크기 | 임상변수 |
|------|---------|---------|--------|-----------|---------|
| **Attia 2019** | Custom ResNet | 12 | 500Hz | 5000x12 | 없음 |
| **Raghunath 2021** | 3-Branch CNN | 15 traces | 500Hz | 분할 입력 | 없음 |
| **Tarabanis 2025** | ResNet-34 | **8** | **250Hz** | **2500x8** | CHARGE-AF 11개 |
| **Khurshid 2022** | CNN + Multi-task | 12 | 500Hz | 5000x12 | CHARGE-AF (Cox) |
| **Brant 2025** | ResNet-based | 12 | 500Hz | - | CHARGE-AF (결합) |

### 6.3 성능 비교

| 연구 | Best AUC | 단독 ECG AUC | F1 | PPV | NPV |
|------|----------|-------------|-----|-----|-----|
| **Attia 2019** | 0.90 (복수) | 0.87 (단일) | 39.2% | - | - |
| **Raghunath 2021** | 0.87 (31일 내) | 0.85 (1년) | - | - | - |
| **Tarabanis 2025** | **0.89** (결합) | 0.83 (단독) | **64%** | **54%** | **94%** |
| **Khurshid 2022** | 0.838 (결합) | 0.823 (단독) | - | - | - |
| **Brant 2025** | 0.85 (결합) | 0.82 (단독) | - | - | - |

### 6.4 학습 하이퍼파라미터 비교

| 연구 | Optimizer | LR | Epochs | Loss | 특이사항 |
|------|-----------|-----|--------|------|---------|
| **Attia 2019** | 미공개 | 미공개 | 미공개 | 미공개 | Supplementary 참조 |
| **Raghunath 2021** | 미공개 | 미공개 | 미공개 | 미공개 | 5-fold CV |
| **Tarabanis 2025** | **Adam** | LR x0.8 감쇠 | **100** | **Cat. CE** | Early stopping 5ep |
| **Khurshid 2022** | 미공개 | 미공개 | 미공개 | Time-to-event | Multi-task 4개 |
| **Brant 2025** | 미공개 | 0.001→0.0001 | 미공개 | 미공개 | Pre-train + Fine-tune |

---

## 7. 왜 작동하는가: 생리학적 메커니즘

### 7.1 심방 리모델링 (Atrial Remodeling)

AF 환자의 심방에서는 NSR 중에도 다음 변화가 ECG에 반영됨:

1. **좌심방 확대 (Left Atrial Enlargement)**: P파 기간 연장, PTFV1 증가
2. **심방 섬유화 (Atrial Fibrosis)**: 전도 지연 → P파 분산(dispersion) 증가
3. **심방간 전도 장애 (Interatrial Block, IAB)**: P파 >120ms, advanced IAB 시 하부리드 이상성 P파
4. **자율신경계 변화**: HRV 패턴 변화

### 7.2 AI가 감지하는 ECG 특징 (Saliency/GradCAM 결과 종합)

| 특징 | 발견 연구 | 설명 |
|------|-----------|------|
| **P파 말단부** | Attia 2019 | 가장 핵심적 예측 영역 |
| **P파 기간 연장** | Khurshid 2022 | 고위험군에서 P파 더 넓음 |
| **QRS 약간 확장** | Khurshid 2022 | 심실 리모델링 동반 가능 |
| **ST 분절 평탄화** | Khurshid 2022 | 미세한 재분극 변화 |
| **P파 주변 전체 영역** | 다수 연구 | P파 onset/offset 포함 |

### 7.3 시간적 진행

Christopoulos 2022의 발견:
- AF 5년 전부터 AI 확률이 점진적으로 상승 (19.8% → 34.0%)
- 이는 심방 리모델링이 **수년에 걸쳐 점진적으로 진행**됨을 시사
- AF 발생 후에도 확률이 계속 상승 (40.9% → 42.2%) → 리모델링 지속

---

## 8. 프로젝트 적용 전략

### 8.1 우리 프로젝트 제약 조건

| 항목 | 우리 | 선행 연구 |
|------|------|-----------|
| **리드 수** | 3-lead (I, II, V1) | 8-12 leads |
| **데이터** | MIMIC-IV ECG (~80만건) | 자체 수집 (16만~65만건) |
| **임상변수** | MIMIC-IV에서 추출 가능 | 자체 EHR |
| **컴퓨팅** | RPi 5 추론 | GPU 클러스터 |

### 8.2 권장 구현 전략

**1순위: Tarabanis 2025 방법론 채택**

근거:
- 8-lead에서 3-lead로 축소가 비교적 용이 (I, II, V1은 이미 포함)
- ResNet-34 아키텍처가 명확하게 기술됨
- CHARGE-AF 결합 방식이 구체적 (concat → FC → softmax)
- Adam optimizer, Cat. CE loss, LR scheduler, early stopping 모두 공개
- MIMIC-IV에서 CHARGE-AF 11개 변수 추출 가능 (기존 매핑 완료)
- +-90일 시간 윈도우로 과제 정의 명확

**구체적 구현 계획**:

```
[입력]
- ECG: 3 x 2,500 (3-lead, 250Hz, 10초)
- 임상: CHARGE-AF 변수 (최소 PMH 4개: DM, MI, HF, 항고혈압제)

[모델]
- 백본: ResNet-34 (1D Conv 버전, 16 residual connections)
- ECG → CNN → 1D representation
- Clinical → FC embedding
- Concat → FC → Softmax (AF/Non-AF)

[학습]
- Optimizer: Adam
- Loss: Categorical cross-entropy
- LR Scheduler: patience 2, factor 0.8
- Early Stopping: patience 5
- Epochs: 최대 100
- 데이터 분할: 7:1:2, 환자 레벨
```

**2순위: Multi-task Learning (Khurshid 2022 참고)**

- 보조 과제 추가: 나이 추정 + 성별 분류 + 현재 AF 감지
- 표현 학습 강화 효과 기대
- 구현 복잡도 증가하지만, MIMIC-IV에서 라벨 즉시 가용

**3순위: Serial ECG (Attia 2019 / Christopoulos 2022 참고)**

- 동일 환자의 복수 ECG 활용 (AUC +0.03 검증됨)
- MIMIC-IV에서 환자당 복수 ECG 존재 → 활용 가능
- 구현: 각 ECG의 AI 확률을 집계 (max, mean, or voting)

### 8.3 데이터 구성 방법 (MIMIC-IV 기반)

```sql
-- 양성 클래스: AF ECG 전후 +-90일 이내 NSR ECG
-- 1) AF ECG 식별: ecg_report_text LIKE '%atrial fibrillation%' OR rhythm_text에서 AF
-- 2) 동일 환자의 NSR ECG 중 AF ECG 날짜 +-90일 이내 필터
-- 3) 180일 이내 복수 AF 시 earliest index AF 기준

-- 음성 클래스: AF 기록이 전혀 없는 환자의 NSR ECG
-- diagnoses_icd에서 AF 관련 ICD 코드 없는 환자만
```

---

## Sources

- [Attia et al. 2019 - PubMed](https://pubmed.ncbi.nlm.nih.gov/31378392/)
- [Attia et al. 2019 - Lancet](https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(19)31721-0/abstract)
- [Raghunath et al. 2021 - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7996054/)
- [Raghunath et al. 2021 - Circulation](https://www.ahajournals.org/doi/full/10.1161/CIRCULATIONAHA.120.047829)
- [Tarabanis et al. 2025 - EHJ Digital Health](https://academic.oup.com/ehjdh/article/6/6/1134/8239531)
- [Khurshid et al. 2022 - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8748400/)
- [Khurshid et al. 2022 - Circulation](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.121.057480)
- [Christopoulos et al. 2023 - JACC:EP](https://www.jacc.org/doi/10.1016/j.jacep.2023.04.008)
- [Brant et al. 2025 - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12569998/)
- [Noseworthy et al. 2022 - Lancet](https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(22)01637-3/abstract)
- [AI-ECG PAF Editorial - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10928874/)
