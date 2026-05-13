# CN-0004: Morphology 3-class ECG 분류기 — 최종 아키텍처

- 작성일: 2026-05-13

## 결정 사항

ECG 다중 진단 분류기로 **Morphology 3-class (Normal/AFib/Other Arrhythmia)** 모델을 최종 채택.
- 구조: CNN-TCN backbone + Projection Head + LogisticRegression (2-stage)
- 학습 방식: SupCon Loss (T=0.1)
- 입력: 5초 ECG (2500 샘플 × 3-lead), 10초 ECG를 앞/뒤 5초로 split → 데이터 2배
- 라벨 정제: 의사 진단 보고서 키워드 매핑 (RR-CV 필터 없음, morphology 학습 강제)

### 성능 (Test, n=7,738)
- **AUROC macro: 0.957** (Normal 0.970, AFib 0.960, OtherArr 0.942)
- Accuracy: 86.3%
- F1 macro: 0.863
- Normal recall: 90.2%, AFib recall: 85.8%, OtherArr recall: 83.0%

### 저장 위치
- Encoder: `G:/AIEKG/ml/checkpoints/morphology-3cls/encoder.pt`
- 분류기: `G:/AIEKG/ml/checkpoints/morphology-3cls/classifier.joblib`
- 결과: `G:/AIEKG/ml/checkpoints/morphology-3cls/train_results.json`

## 배경/맥락

기존 SNN(SupCon) 학습이 RR-CV 임계값 기반 라벨 정제(Normal<5%, AFib>15%)를 사용하여 모델이 RR shortcut에 편향되어 학습됨. 사용자가 "RR 간격이 아닌 파형 패턴(PQRST morphology) 자체를 학습"하길 원함.

추가로 단일 모델로 여러 질환(Normal/AFib/OtherArr/HK)을 동시 분류하는 요구사항이 있었음.

## 검토한 대안

### 대안 1: 4-class (Normal/AFib/OtherArr/Hyperkalemia) — 기각

- 클래스 균형: 각 8,515개 (HK 데이터 한계)
- 학습 데이터: K<5.0 + time_diff<1hr 필터, 5초 split
- **결과**: AUROC macro 0.906, Acc 71.1%
  - AFib AUROC 0.966 ✅
  - OtherArr AUROC 0.953 ✅
  - Normal AUROC **0.855**, recall **60.8%** ⚠️
  - HK AUROC **0.797**, recall **44.2%** ⚠️
- **기각 사유**: Normal과 HK가 서로 32~38% 혼동 발생. T파 morphology 만으로는 Hyperkalemia 분리 부족.

### 대안 2: 4-class + 임상 변수 (나이, HTN, DM) — 기각

- ECG 임베딩(128) + 나이 + 고혈압 + 당뇨 = 131-dim
- **결과**: AUROC macro 0.906 → 0.906 (변화 없음)
- 가중치 측면은 의학적으로 합리적 (HTN→AFib +0.239)이지만 ECG 신호가 이미 강해서 추가 정보 효과 미미
- **기각 사유**: 임상 변수 추가가 HK의 핵심 약점을 해결 못함 (eGFR 같은 결정적 변수 부재)

### 대안 3: 3-class (HK 제외) — 최종 채택

- 클래스: Normal/AFib/OtherArr 각 8,515개
- 동일 학습 방식
- **결과**: AUROC macro **0.957** (4-class 0.906에서 +0.051)
  - Normal recall **60.8% → 90.2%** (+29.4%p) 🎯
  - AFib/OtherArr는 거의 동일하게 유지
- **채택 사유**: HK 클래스가 Normal과의 경계를 흐리게 만들고 있었음. HK 제거로 모든 클래스 성능 향상 또는 유지.

## 근거

1. **HK는 ECG morphology만으로는 한계**
   - T파 형태 변화가 RR 불규칙성보다 미세한 신호
   - K 5.5~6.0 경계 케이스가 다수 (라벨 신뢰도 낮음)
   - eGFR/Cr/RAAS 억제제 같은 임상 정보가 결정적 변수 (ECG 단독 부족)

2. **3-class는 ECG만으로 충분히 우수**
   - AFib: RR + P파 + fibrillation wave가 명확한 신호
   - OtherArr: PVC/PAC/flutter 등 morphology 특징 학습 가능
   - Normal: 다른 두 클래스의 부재로 잘 구분됨

3. **morphology 학습 강제 효과 검증**
   - RR-CV 필터 제거 → 다양한 RR 분포의 ECG 학습
   - AUROC가 OtherArr에서 0.834 → 0.953 (+0.119) 큰 폭 향상
   - 모델이 RR shortcut 못 쓰고 진짜 morphology 학습한 증거

4. **5초 split의 효과**
   - 데이터 2배 (34K → 68K)
   - 위치 의존성 줄임 (모델이 어디서 잘리든 패턴 학습)
   - augmentation 아닌 단순 split (사용자 명시 요구사항)

5. **설명가능성 우수 (의료기기 관점)**
   - SupCon 임베딩 → t-SNE 시각화 가능
   - 임베딩 거리 기반 "유사 사례 검색" 가능
   - LogisticRegression 가중치 직접 해석 가능
   - End-to-end CNN 대비 의사/규제기관 신뢰 확보 용이

## 영향/주의사항

### 후속 작업 필요
- **HK 감지는 별도 모델로 분리**
  - eGFR/크레아티닌, RAAS 억제제 복용 여부 등 임상 변수 필수
  - MIMIC-IV labevents에서 추출 가능
  - 추후 작업: `train_hyperkalemia_separate.py` (eGFR + 약물 + ECG 임베딩)

### 데이터 제약
- 현재 학습 데이터는 **K<5.0 + time_diff<1hr** 필터로 25.5K (균형)
- 더 큰 데이터셋 (필터 없이 470K Normal + 90K AFib + 215K OtherArr)로 재학습 시 정확도 더 올라갈 가능성
- 추후 작업: 전체 데이터로 학습 시도

### CNN-TCN backbone 활용
- 기존 부정맥 분류 (cnn-tcn-3lead, Acc 90.6%) 모델의 backbone 구조 재사용
- 사전학습 가중치는 사용 안 함 (from-scratch SupCon 학습이 더 적합)

### 임상 변수의 효과
- 나이/HTN/DM 추가 효과는 미미 (+0.0008 AUROC)
- 그러나 LR 가중치는 의학적으로 합리적 방향 (해석가능성 가치는 유지)
- 향후 eGFR 추가 시 HK 모델에서 효과 클 것

### 입력 형식 주의
- 5초 = 2500 샘플, 500Hz 가정
- 3-lead: Lead I, Lead II, Lead III(=II-I 계산)
- 추론 시 10초 ECG는 앞/뒤 5초로 잘라 2번 추론 후 평균 권장
