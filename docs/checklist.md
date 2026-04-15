# AIEKG 프로젝트 체크리스트 (공정표)

> 기획서(설계도)를 따라가며 진행 상황을 추적.
> ✅ 완료 / 🔄 진행중 / ⬚ 미착수 / ⏸️ 보류 / ❌ 기각

---
마지막 업데이트: 2026-04-07

## Phase 0: 개발 환경 & 인프라 구축

- ✅ ESP32 개발환경 선정 (ESP-IDF / PlatformIO)
- ✅ 프로젝트 저장소 생성
- ✅ Claude Code 에이전트 시스템 구축
  - ✅ senior-code-architect 에이전트
  - ✅ test-diagnose-runner 에이전트
  - ✅ ekg-project-planner 에이전트
- ✅ 훅 시스템 구축
  - ✅ UserPromptSubmit: prompt-analyzer (선배의 귀띔)
  - ✅ UserPromptSubmit: worklog-writer (작업일지)
  - ✅ PostToolUse: code-review-reminder (코드 리뷰 귀띔)
  - ✅ PostToolUse: context-note-reminder (맥락노트 리마인더)
- ✅ 스킬북 챕터 시스템 (ch01~ch10)
- ✅ 맥락노트(시방서) 구조 생성 + 첫 노트(CN-0001) 작성
- ✅ 체크리스트(공정표) 생성
- ⬚ 프로젝트 기획서(설계도) 정식 작성

## Phase 1: SPI 통신 검증 (Gate 1)

> 목표: ADS1292R 디바이스 ID(0x73) 읽기 성공

- ⬚ ADS1292R 부품 구매/수령
- ⬚ ESP32 + ADS1292R 브레드보드 배선
  - ⬚ SPI 핀 연결 (MOSI/MISO/SCLK/CS)
  - ⬚ DRDY 인터럽트 핀 연결
  - ⬚ START/RESET 핀 연결
  - ⬚ 전원 연결 (3.3V → LDO → ADS 아날로그/디지털)
- ⬚ SPI 드라이버 구현
  - ⬚ SPI 버스 초기화
  - ⬚ ADS1292R 레지스터 읽기/쓰기
  - ⬚ 디바이스 ID 읽기 (0x73 확인)
  - ⬚ 초기 레지스터 설정 (CONFIG1, CONFIG2, CH1SET, CH2SET)
- ⬚ 내부 테스트 신호 확인
  - ⬚ 1mV 1Hz 구형파 출력
  - ⬚ 시리얼 플로터로 파형 확인
- ⬚ **Gate 1 판정**: SPI 통신 성공 ✅/❌

## Phase 2: 신호 취득 (Gate 2)

> 목표: 실제 인체에서 P-QRS-T 식별 가능

- ⬚ 전극 연결 (3전극: RA, LA, RL)
- ⬚ 리드오프 감지 구현
- ⬚ RLD (Right Leg Drive) 활성화
- ⬚ Raw ECG 데이터 수집
- ⬚ 디지털 필터 구현
  - ⬚ 고역통과 필터 (0.5Hz, DC 제거)
  - ⬚ 노치 필터 (60Hz 전원 노이즈)
  - ⬚ 저역통과 필터 (40Hz)
- ⬚ 시리얼 플로터로 필터링된 ECG 파형 확인
- ⬚ 기저선 안정성 검증
- ⬚ **Gate 2 판정**: P-QRS-T 식별 가능 ✅/❌

## Phase 3: 실시간 처리 (Gate 3)

> 목표: 500SPS 연속 처리 + BLE 전송 안정적

- ⬚ QRS 검출 알고리즘 구현 (Pan-Tompkins)
  - ⬚ 대역통과 필터 (5-15Hz)
  - ⬚ 미분 + 제곱 + 이동평균
  - ⬚ 적응형 임계값
  - ⬚ R-R 간격 검증
- ⬚ 심박수 계산 (BPM)
- ⬚ BLE 서비스 구현
  - ⬚ ECG 데이터 Notify
  - ⬚ Heart Rate Notify
  - ⬚ 리드 상태 Notify
- ⬚ 링버퍼 구현 (실시간 스트리밍)
- ⬚ FreeRTOS 태스크 분리 (센서/처리/통신)
- ⬚ 성능 벤치마크
  - ⬚ ADC 읽기 지연 < 2ms
  - ⬚ 필터 처리 < 1ms/sample
  - ⬚ 총 파이프라인 < 50ms
  - ⬚ 힙 메모리 < 100KB
- ⬚ 24시간 연속 안정성 테스트
- ⬚ **Gate 3 판정**: 실시간 처리 안정적 ✅/❌

## Phase 4: AI/ML 모델 개발 (Gate 4)

> 목표: 3가지 핵심 AI 기능 완성

### 4-1. 데이터 수집 및 전처리 (완료)
- ✅ MIMIC-IV ECG 데이터 수집 (775,367건, 159,666명)
- ✅ HDF5 전처리 (ecg_preprocessed.h5, 173GB)
- ✅ 3-lead 변환 (Lead I, II + Lead III = II - I)
- ✅ 학습/검증/테스트 분할

### 4-2. 부정맥 분류 모델 (완료)
- ✅ CNN-TCN 아키텍처 설계 (126,355 params)
- ✅ 3-lead 학습 + 검증
- ✅ Test Accuracy 90.6% 달성
- ✅ 체크포인트 저장 (checkpoints/cnn-tcn-3lead/best_model.pt)

### 4-3. 발작성 AF 감지 모델 (완료, 2026-04-07)
- ✅ AF 진단 환자 목록 추출 (MIMIC-IV diagnoses_icd)
- ✅ 정상 리듬 ECG 필터링 (AF 환자의 sinus rhythm ECG)
- ✅ Negative 샘플 구성 (AF 이력 없는 환자의 sinus rhythm ECG)
- ✅ 데이터셋 구축 + HDF5 저장 (52만건: 양성 52,202 / 음성 471,157)
- ❌ CNN-TCN 백본 Transfer Learning → AUROC 0.68 정체, 기각
- ✅ ResNet-34 from scratch (ECG only) → AUROC 0.8038
- ✅ ResNet-34 + 임상정보 (ECG + Clinical) → **AUROC 0.8240** (최종 모델)
- ✅ 임계값 최적화: 0.61 (Sens 60%, Spec 85%, Prec 31%) 권장
- ⬚ 목표: AUROC >= 0.83 → **미달 (0.82)**, 3-lead 한계 내 논문 ECG only(0.83)와 근접, 수용 가능
- ✅ 체크포인트 저장 (checkpoints/paroxysmal-af-resnet34-clinical/)

### 4-4. AFib 예측 모델 개선 (우선순위 2)
- ⬚ MIMIC-IV diagnoses_icd에서 임상정보 추출
  - ⬚ 당뇨(DM) 플래그
  - ⬚ 심부전(HF) 플래그
  - ⬚ 심근경색(MI) 플래그
  - ⬚ 항고혈압제(AHT) 사용 플래그
- ⬚ 시퀀스 TCN + 임상 변수 결합 아키텍처 설계
- ⬚ 학습 + 검증
- ⬚ 목표: AUROC >= 0.80 (현재 0.7355)
- ⬚ 체크포인트 저장

### 4-5. 배포 준비
- ⬚ 모델 3종 ONNX 변환
- ⬚ Raspberry Pi 5 추론 벤치마크

- ⬚ **Gate 4 판정**: 모델 3종 성능 목표 달성 ✅/❌

## Phase 5: 소프트웨어 개발 (Gate 5)

> 목표: Raspberry Pi 5 추론 서버 + UI 대시보드 + 부가 기능

- ⬚ BLE 수신 + 데이터 파이프라인 (Raspberry Pi 5)
- ⬚ AI 추론 서버 (모델 3종 ONNX 순차 추론)
- ⬚ PostgreSQL DB 저장 (측정별 ECG + AI 분석 결과)
- ⬚ 5인치 LCD UI
  - ⬚ 실시간 ECG 파형 표시
  - ⬚ 심박수/HRV 표시
  - ⬚ 부정맥 분류 결과 표시
  - ⬚ 발작성 AF 감지 확률 표시
  - ⬚ AF 예측 위험도 표시
- ⬚ 위험도 대시보드
  - ⬚ AF 감지 확률 추이 그래프
  - ⬚ 측정 이력 (날짜별 ECG 기록)
- ⬚ 병원 전송용 PDF 리포트 (ECG 파형 + AI 분석 결과)
- ⬚ 경고/알림 기능 ("병원 방문 권장" 수준)
- ⬚ 전자청진기 데이터 연동
- ⬚ **Gate 5 판정**: AI 추론 지연 N초 이내 ✅/❌

## Phase 6: 시스템 통합 및 검증

> 목표: HW + FW + SW + AI 전체 통합

- ⬚ HW + FW + SW 통합 테스트
- ⬚ AI 모델 3종 엔드투엔드 검증
- ⬚ 실측 데이터 정확도 테스트 (학습 vs 실측 차이)
- ⬚ 전자청진기 연동 통합 테스트
- ⬚ 장시간 안정성 테스트 (24시간 연속 동작)

## Phase 7: 마무리

> 목표: 완성된 프로토타입

- ⬚ PCB 설계 (KiCad)
  - ⬚ 회로도 작성
  - ⬚ PCB 레이아웃
  - ⬚ 거버 파일 생성 / 제작 발주
- ⬚ 케이스 설계 (3D 프린팅)
- ⬚ 장시간 안정성 테스트 (48시간+)
- ⬚ 최종 문서화
  - ⬚ 하드웨어 설계 문서
  - ⬚ 펌웨어 API 문서
  - ⬚ 사용자 매뉴얼
- ⬚ 오픈소스 공개 준비

---

## 다음 할 일 (Next Actions)

1. **[우선순위 1] Phase 4-4**: AFib 예측 모델 개선 (임상정보 추출 + 결합, 목표 AUROC 0.80+)
2. **[우선순위 2] Phase 4-5**: 모델 3종 ONNX 변환 + Raspberry Pi 5 추론 벤치마크
3. Phase 1 시작: ADS1292R 부품 구매 -> 브레드보드 배선 -> SPI 드라이버 구현 (하드웨어 도착 후)

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|-----------|
| 2026-03-29 | 체크리스트 초기 생성. Phase 0 인프라 작업 완료 반영. |
| 2026-04-06 | AI 기능 3종 재정의 반영. Phase 4를 ML 모델 개발로 변경, Phase 5를 소프트웨어 개발로 변경. 부정맥 분류 완료 반영, 발작성 AF 감지(신규) + AFib 예측 개선 항목 추가. |
| 2026-04-07 | 발작성 AF 감지 모델 완료 반영 (AUROC 0.8240). CNN-TCN TL 실패/기각, ResNet-34+Clinical 최종 채택. 다음 할 일 우선순위 조정. |
