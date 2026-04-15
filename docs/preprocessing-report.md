# ECG 전처리 파이프라인 보고서

**작성일**: 2026-04-01  
**상태**: 파이프라인 구현 완료, 샘플 테스트 통과

---

## 1. 개요

MIMIC-IV ECG 데이터(80만건)를 CNN-TCN 부정맥 분류 모델 학습용으로 전처리하는 파이프라인을 구현했다.

## 2. 원본 데이터 현황

| 항목 | 값 |
|------|-----|
| 총 ECG 레코드 | 800,035건 |
| 포맷 | WFDB (.dat/.hea) |
| 리드 수 | 12-lead |
| 샘플링 레이트 | 500Hz |
| 길이 | 10초 (5,000 samples) |
| 고유 report_0 라벨 | 1,570종 |

## 3. 라벨 매핑 (3-class)

report_0 텍스트를 키워드 기반으로 3개 클래스로 매핑했다.  
마침표/대소문자 변형을 모두 처리하기 위해 소문자 변환 + strip 후 키워드 매칭 방식을 사용했다.

### 매핑 규칙
- **Class 0 (Normal)**: "sinus rhythm", "normal sinus rhythm", "sinus arrhythmia" 포함
- **Class 1 (AFib)**: "atrial fibrillation", "atrial flutter" 포함 (우선 매칭)
- **Class 2 (Other)**: 위에 해당하지 않는 나머지

### 제외 기준
- report_0 결측 (1건)
- 데이터 품질 경고 텍스트 포함: "data quality", "pacer detection suspended", "pediatric criteria", "age not entered"

### 매핑 결과

| 클래스 | 건수 | 비율 |
|--------|------|------|
| 0 (Normal) | 475,232 | 60.5% |
| 1 (AFib) | 91,539 | 11.7% |
| 2 (Other) | 218,903 | 27.9% |
| 제외 | 14,361 | - |
| **유효 합계** | **785,674** | **100%** |

**클래스 불균형 참고**: Normal이 60%로 다수. 학습 시 class weight 또는 oversampling 필요.

## 4. 피처 구성

### 4.1 파형 피처 (메인 입력)
- **형태**: (5000, 12) float32
- **전처리 순서**:
  1. Bandpass filter (0.5-50Hz, Butterworth 4차)
  2. 60Hz Notch filter (Q=30)
  3. Z-score 정규화 (리드별)
- **품질 검사**: NaN/Inf 체크, flat line(std < 1e-6) 검출

### 4.2 수치 피처 (보조 입력)
machine_measurements 테이블에서 추출:

| 피처 | 설명 | 계산 |
|------|------|------|
| rr_interval | RR 간격 (ms) | 직접 사용 |
| qrs_duration | QRS 지속시간 (ms) | qrs_end - qrs_onset |
| p_duration | P파 지속시간 (ms) | p_end - p_onset |
| qrs_axis | QRS 전기축 (도) | 직접 사용 |

### 4.3 환자 피처 (보조 입력)
patients 테이블에서 추출:

| 피처 | 설명 | 처리 |
|------|------|------|
| age | 나이 | anchor_age 직접 사용 |
| gender_code | 성별 | M=0, F=1 |

## 5. 출력 포맷

### HDF5 파일 구조 (`ml/data/ecg_preprocessed.h5`)

| 데이터셋 | Shape | Dtype | 설명 |
|----------|-------|-------|------|
| waveform | (N, 5000, 12) | float32 | 전처리된 12-lead 파형 |
| label | (N,) | int8 | 3-class 라벨 |
| numeric_features | (N, 4) | float32 | 수치 피처 4개 |
| patient_features | (N, 2) | float32 | 환자 피처 2개 |
| subject_id | (N,) | int64 | 환자 ID |
| study_id | (N,) | int64 | 검사 ID |

- 압축: gzip level 4 (waveform만)
- 메타데이터: sample_rate, lead_names, class_names 등 attrs에 저장

## 6. 테스트 결과

### 100건 테스트
| 항목 | 결과 |
|------|------|
| 성공 | 99건 (99.0%) |
| 품질 불량 제외 | 1건 |
| 파일 크기 | 21.2 MB |

### 1,000건 테스트
| 항목 | 결과 |
|------|------|
| 성공 | 990건 (99.0%) |
| 품질 불량 제외 | 10건 (1.0%) |
| 파일 크기 | 211.3 MB |
| 처리 속도 | ~66건/초 |

### 1,000건 클래스 분포
| 클래스 | 건수 | 비율 |
|--------|------|------|
| Normal | 598 | 60.4% |
| AFib | 100 | 10.1% |
| Other | 292 | 29.5% |

## 7. 전체 실행 예상

| 항목 | 예상값 |
|------|--------|
| 전체 대상 | ~785,620건 |
| 예상 성공 | ~777,000건 (99%) |
| 예상 소요 시간 | ~3.3시간 (66건/초 기준) |
| 예상 파일 크기 | ~165 GB (비압축 기준) |

**주의**: 165GB는 디스크 용량 확인 필요. 리드 수를 줄이거나 배치 분할 저장 고려.

## 8. 파일 구조

```
g:/AIEKG/ml/
├── preprocessing/
│   ├── config.py              # 전처리 설정값
│   ├── label_mapper.py        # report_0 → 3-class 매핑
│   ├── ecg_filter.py          # 필터링 + 정규화
│   ├── patient_features.py    # 환자 피처 추출
│   ├── analyze_labels.py      # 라벨 분포 분석 (단독 실행)
│   └── run_pipeline.py        # 메인 파이프라인
└── data/
    └── ecg_preprocessed.h5    # 출력 파일
```

## 9. 사용법

```bash
# 테스트 (100건)
python run_pipeline.py --max-samples 100

# 전체 실행
python run_pipeline.py

# DB에서 환자 피처 로드
python run_pipeline.py --use-db
```

## 10. 다음 단계

1. **디스크 용량 확인** 후 전체 실행 또는 리드 선택 (2~3 lead)
2. **train/val/test 분할** 스크립트 추가 (환자 단위 분할)
3. **CNN-TCN 모델** 구현 및 학습 시작
4. **클래스 불균형 처리**: class weight 또는 oversampling 적용
