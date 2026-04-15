# CN-0003: 발작성 AF 감지 모델 — 아키텍처 선정 및 결과

- 작성일: 2026-04-07

## 결정 사항

발작성 AF 감지 모델로 **ResNet-34 (1D Conv, from scratch) + 임상정보 Late Fusion**을 최종 채택하였다.
- Test AUROC: 0.8240
- 권장 임계값: 0.61 (Sens 60%, Spec 85%, Prec 31%)

## 배경/맥락

- 기획서 v3.0에서 발작성 AF 감지를 AI 기능 3종 중 우선순위 1로 설정
- 원래 계획: 기존 부정맥 분류용 CNN-TCN 백본을 Transfer Learning으로 재활용
- 참고 논문: Tarabanis et al. (EHJ Digital Health 2025) -- 8-lead + CHARGE-AF로 AUROC 0.89

## 검토한 대안

### 대안 1: CNN-TCN Transfer Learning (기각)
- 부정맥 분류 모델(CNN-TCN 126K params) backbone frozen -> 이진분류 헤드 교체
- **결과: AUROC 0.68에서 정체**
- 원인: 부정맥 분류(정상/AFib/Other 구분)와 발작성 AF 감지(정상 리듬 내 미세 AF 흔적)는 학습하는 피처가 근본적으로 다름

### 대안 2: ResNet-34 ECG only (채택 후 확장)
- ResNet-34 (7.3M params) from scratch, 3-lead x 5000 samples
- **결과: AUROC 0.8038**
- 대형 모델이 52만건 데이터에서 충분히 학습

### 대안 3: ResNet-34 + 임상정보 (최종 채택)
- ResNet-34 CNN 512d + tabular(DM/HF/MI/AHT + numeric + patient) Late Fusion
- **결과: AUROC 0.8240 (+0.02 향상)**
- 임상정보(ECG 외부 정보) 결합 효과 재확인

## 근거

1. **Transfer Learning 한계**: 부정맥 분류와 발작성 AF 감지는 다른 과제. 소규모 백본(126K)으로는 미세한 AF 흔적 학습 불가
2. **대형 모델 정당성**: 52만건 데이터 규모에서 7.3M params ResNet-34는 과적합 없이 수렴
3. **임상정보 효과**: ECG 파형에 없는 외부 정보(병력)가 +0.02 AUROC 향상 기여 -- 기존 HRV/Intervals 추가가 실패한 것과 대조적
4. **3-lead 한계 수용**: 목표 AUROC 0.83 미달(0.82)이나, 논문 ECG only(0.83)와 거의 동등. V1 없이 Lead II P파 형태만으로 학습 성공

## 영향/주의사항

1. **모델 크기**: 7.3M params는 기존 부정맥 분류(126K) 대비 58배. Raspberry Pi 5 ONNX 추론 시 지연 시간 확인 필요
2. **임계값 선택**: 배포 시 용도별 임계값 조정 필요 (스크리닝 vs 높은 정밀도)
3. **AUROC 0.82 한계**: 8-lead나 12-lead 장비에서는 더 높은 성능 가능하나, 본 프로젝트 3-lead 제약 내 최대치
4. **AFib 예측 모델 개선 시**: 발작성 AF에서 확인된 임상정보 효과(+0.02)를 AFib 예측에도 적용할 근거 확보
