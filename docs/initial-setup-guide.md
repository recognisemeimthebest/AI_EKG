# Claude Code 개발 인프라 초기 세팅 가이드

> 이 문서는 Claude Code의 에이전트 + 훅 + 스킬북 + 문서 체계를 **처음부터 구축하는 방법**을 설명합니다.
> 다른 프로젝트에 이 시스템을 복제/적용할 때 이 가이드를 따르세요.
>
> 원본 프로젝트: AIEKG (AI 전자청진기 + EKG 장치)
> 최초 작성일: 2026-03-29

---

## 목차

1. [개요](#1-개요)
2. [전체 아키텍처](#2-전체-아키텍처)
3. [디렉토리 구조](#3-디렉토리-구조)
4. [에이전트 시스템](#4-에이전트-시스템)
5. [훅 시스템](#5-훅-시스템)
6. [스킬북 챕터 시스템](#6-스킬북-챕터-시스템)
7. [3종 문서 체계](#7-3종-문서-체계)
8. [작업일지 시스템](#8-작업일지-시스템)
9. [새 프로젝트에 적용하는 방법](#9-새-프로젝트에-적용하는-방법)
10. [커스터마이징 가이드](#10-커스터마이징-가이드)

---

## 1. 개요

### 이 시스템이 뭔가?

Claude Code를 "그냥 대화형 AI"가 아닌 **프로젝트 전담 개발팀**으로 활용하기 위한 인프라다. 사용자 프롬프트를 분석해서 적절한 전문가(에이전트)를 추천하고, 코드 작성 시 자동으로 위험 요소를 체크하며, 모든 작업 기록을 자동으로 남긴다.

### 왜 만들었나?

| 문제 | 해결 |
|------|------|
| 영역이 넓은 프로젝트에서 Claude가 모든 맥락을 한번에 로드하면 토큰 낭비 | 에이전트별 전문 역할 분리 + 스킬북 on-demand 로딩 |
| 코드 리뷰 없이 작성만 하면 위험 요소 놓침 | PostToolUse 훅으로 자동 코드 리뷰 귀띔 |
| 며칠 비우면 "어디까지 했더라?" 파악 어려움 | 작업일지 자동 기록 + "이어서" 감지 |
| 의사결정 근거가 사라짐 (머릿속에만 있음) | 3종 문서 체계로 결정 근거 영구 보존 |
| 프롬프트마다 "이건 코드 에이전트한테 시켜야 하는데"를 직접 판단해야 함 | prompt-analyzer 훅이 자동으로 추천 |

### 핵심 컨셉: 건설 현장 비유

```
기획서(설계도) = "뭘 만들 건가?"    → docs/plan/
맥락노트(시방서) = "왜 이렇게 했나?"  → docs/context-notes/
체크리스트(공정표) = "어디까지 했나?"  → docs/checklist.md
```

설계도만 있으면 "왜 이 자재?"를 모르고, 시방서만 있으면 "어디까지 진행됐나?"를 모른다. 세 문서가 서로 보완해야 프로젝트 추적이 완전해진다.

---

## 2. 전체 아키텍처

### 시스템 구성도

```
사용자 프롬프트 입력
       |
       v
+-------------------------------+
| UserPromptSubmit 훅 (2개)     |
|                               |
| 1. prompt-analyzer.sh         |
|    - 키워드 감지              |
|    - 에이전트 추천            |  ---> stdout: 추천 메시지
|    - 스킬북 챕터 추천         |
|                               |
| 2. worklog-writer.sh          |
|    - 오늘 날짜 일지에 기록    |  ---> .claude/worklog/YYYY-MM-DD.md
+-------------------------------+
       |
       v
+-------------------------------+
| Claude Code 본체              |
|                               |
|  에이전트 (서브에이전트)       |
|  +---------------------------+|
|  | senior-code-architect     ||  <--- 코드 작성/리뷰/리팩토링
|  | test-diagnose-runner      ||  <--- 테스트/디버깅/진단
|  | ekg-project-planner       ||  <--- 기획/문서/보고
|  +---------------------------+|
|                               |
|  각 에이전트는 필요 시         |
|  스킬북 챕터를 Read로 로드    |
|  +---------------------------+|
|  | .claude/skillbook/        ||
|  | ch01 ~ ch10               ||
|  +---------------------------+|
+-------------------------------+
       |
       | Write / Edit / Bash 도구 사용
       v
+-------------------------------+
| PostToolUse 훅 (1개)          |
|                               |
| code-review-reminder.sh       |
|  - 위험한 작업 감지           |
|  - 에러 처리 누락 체크        |  ---> JSON: additionalContext
|  - 보안 체크                  |
|  - 임베디드 특화 체크         |
|  - 맥락노트/체크리스트 리마인더|
+-------------------------------+
       |
       v
+-------------------------------+
| 문서 체계 (3종)               |
|                               |
| docs/plan/          설계도    |
| docs/context-notes/ 시방서    |
| docs/checklist.md   공정표    |
+-------------------------------+
```

### 데이터 흐름 요약

1. **입력 시**: 프롬프트 -> prompt-analyzer(에이전트/챕터 추천) + worklog-writer(일지 기록)
2. **작업 중**: 에이전트가 스킬북 챕터를 on-demand Read
3. **출력 시**: Write/Edit/Bash 사용 -> code-review-reminder(코드 리뷰 귀띔)
4. **기록**: 의미 있는 결정은 맥락노트에, 진행 상황은 체크리스트에 반영

---

## 3. 디렉토리 구조

```
프로젝트루트/
|
+-- .claude/
|   +-- settings.local.json          # 훅 등록 + 권한 설정
|   |
|   +-- agents/                       # 서브에이전트 정의
|   |   +-- senior-code-architect.md  # 코드 전문가
|   |   +-- test-diagnose-runner.md   # 테스트 전문가
|   |   +-- ekg-project-planner.md    # 기획 전문가
|   |
|   +-- hooks/                        # 훅 스크립트
|   |   +-- prompt-analyzer.sh        # 프롬프트 분석 -> 추천
|   |   +-- worklog-writer.sh         # 작업일지 자동 기록
|   |   +-- code-review-reminder.sh   # 코드 리뷰 귀띔
|   |
|   +-- skillbook/                    # 도메인별 상세 가이드
|   |   +-- _index.md                 # 챕터 목차
|   |   +-- ch01-esp32-embedded.md    # ESP32 임베디드
|   |   +-- ch02-signal-ekg.md        # 신호처리/EKG
|   |   +-- ch03-security.md          # 보안
|   |   +-- ch04-error-handling.md    # 에러 처리
|   |   +-- ch05-ml-ai.md            # AI/ML
|   |   +-- ch06-communication.md     # 통신
|   |   +-- ch07-hardware.md          # 하드웨어
|   |   +-- ch08-testing.md           # 테스트
|   |   +-- ch09-planning.md          # 기획
|   |   +-- ch10-code-style.md        # 코드 스타일
|   |
|   +-- worklog/                      # 자동 생성되는 작업일지
|   |   +-- 2026-03-29.md             # 날짜별 일지
|   |
|   +-- agent-memory/                 # 에이전트별 영구 기억
|       +-- senior-code-architect/
|       +-- test-diagnose-runner/
|       +-- ekg-project-planner/
|
+-- docs/
    +-- plan/                          # 기획서 (설계도)
    |   +-- project-plan-v1.0.md
    |
    +-- context-notes/                 # 맥락노트 (시방서)
    |   +-- _index.md                  # 노트 목록
    |   +-- CN-0001-dev-infra-design.md
    |
    +-- checklist.md                   # 체크리스트 (공정표)
```

### 각 파일의 역할

| 파일 | 역할 |
|------|------|
| `settings.local.json` | 훅 등록(어떤 이벤트에 어떤 스크립트 실행), 권한 설정(허용할 도구/도메인) |
| `agents/*.md` | 에이전트의 전문성, 발동 조건, 참조할 스킬북 챕터를 정의 |
| `hooks/*.sh` | 이벤트 발생 시 자동 실행되는 bash 스크립트 |
| `skillbook/*.md` | 도메인별 상세 가이드. 에이전트가 필요할 때만 Read로 로드 |
| `worklog/*.md` | 날짜별 자동 생성되는 작업 타임라인 |
| `agent-memory/` | 에이전트가 대화 간 유지하는 영구 기억 (메모리 시스템) |
| `docs/plan/` | 프로젝트 전체 로드맵, 목표, 범위 |
| `docs/context-notes/` | 의사결정 근거 기록 ("왜 이렇게?") |
| `docs/checklist.md` | Phase별 진행 상황 추적 |

---

## 4. 에이전트 시스템

### 3개 에이전트 개요

| 에이전트 | 역할 | 모델 | 색상 | 메모리 범위 |
|---------|------|------|------|------------|
| `senior-code-architect` | 코드 작성/리뷰/리팩토링 | opus | blue | project |
| `test-diagnose-runner` | 테스트/디버깅/진단 | opus | green | project |
| `ekg-project-planner` | 기획/문서/보고 | opus | red | project |

### 발동 조건 (prompt-analyzer가 감지)

**senior-code-architect**:
- 코드, 작성, 구현, 드라이버, 함수, 클래스, 모듈, 리팩토링, spi, i2c, gpio, firmware, esp32, build, cmake 등
- 파일 패턴: `.c`, `.cpp`, `.h`, `.py`, `.js`, `.ts`
- 코드 패턴: `#include`, `void`, `typedef`, `struct`, `#define`, `esp_err_t`

**test-diagnose-runner**:
- 테스트, test, 디버그, debug, 에러, error, 버그, bug, 크래시, crash, 실패, fail, 확인해, 검증, verify, 로그, log
- 에러 메시지 패턴: `error:`, `warning:`, `traceback`, `segfault`, `undefined reference`

**ekg-project-planner**:
- 계획, 일정, 스케줄, 마일스톤, 로드맵, 보고서, 문서, 진행 상황, 리스크, 예산, bom, phase, 단계

### 에이전트 파일 구조 (실제 예시)

에이전트 파일은 `.claude/agents/` 디렉토리에 위치하며, YAML frontmatter + Markdown body로 구성된다.

```markdown
---
name: senior-code-architect
description: "Use this agent when writing new code, reviewing existing code,
fixing errors, or improving code structure. ...

Examples:
- User: \"ESP32에서 SPI 드라이버를 작성해줘\"
  Assistant: \"senior-code-architect 에이전트로 작성하겠습니다.\""
model: opus
color: blue
memory: project
---

You are a senior software architect with 15+ years of experience...

## Core Responsibilities
1. **Code Writing** -- Clean, SOLID, testable code with proper error handling
2. **Code Review** -- Logic errors, memory/resource leaks, race conditions
3. **Structure Improvement** -- Refactoring, design patterns, module boundaries

## Skillbook (상세 가이드)
작업에 필요한 챕터를 Read tool로 열어서 참고하세요:

| 상황 | 챕터 파일 |
|------|-----------|
| ESP32/SPI/I2C/GPIO/FreeRTOS | `.claude/skillbook/ch01-esp32-embedded.md` |
| ECG 신호처리/필터/QRS | `.claude/skillbook/ch02-signal-ekg.md` |
| ... | ... |

**prompt-analyzer 훅이 관련 챕터를 추천합니다.** 추천된 챕터를 우선 참고하세요.

## Mandatory: 변경 기록
코드 작성/수정 시 반드시 한국어로 출력:
## 변경 기록
### 발견 사항
### 수정 내용
### 판단 근거
### 주의 사항

## Quality Self-Check
- [ ] 컴파일/실행 에러 없음
- [ ] 에러 처리 완비
- [ ] 네이밍 일관성
- [ ] 변경 기록 작성 완료
```

### frontmatter 필드 설명

| 필드 | 설명 | 값 |
|------|------|----|
| `name` | 에이전트 이름 (Agent tool 호출 시 사용) | 영문 kebab-case |
| `description` | 발동 조건 설명. Claude가 이 텍스트를 읽고 언제 이 에이전트를 쓸지 판단 | 상세한 예시 포함 권장 |
| `model` | 사용할 모델 | `opus`, `sonnet` 등 |
| `color` | UI 표시 색상 | `blue`, `green`, `red` 등 |
| `memory` | 메모리 범위 | `project` (프로젝트 공유) 또는 생략 |

### 에이전트 body에 반드시 포함할 것

1. **정체성 선언** -- "You are a ..."
2. **핵심 역할/원칙** -- 3개 이내로 압축
3. **스킬북 참조 테이블** -- 어떤 상황에 어떤 챕터를 읽을지
4. **출력 형식** -- 에이전트가 반드시 따를 보고/기록 형식
5. **자기 검증 체크리스트** -- 완료 전 확인 항목

---

## 5. 훅 시스템

### 훅이란?

Claude Code가 특정 이벤트 시점에 자동으로 실행하는 bash 스크립트다. 사용자가 명시적으로 호출하지 않아도 동작한다.

### 훅 등록 (settings.local.json)

```json
{
  "permissions": {
    "allow": [
      "Bash(프로젝트루트/.claude/hooks/*)",
      "Bash(bash \"프로젝트루트/.claude/hooks/prompt-analyzer.sh\")",
      "Bash(bash \"프로젝트루트/.claude/hooks/code-review-reminder.sh\")"
    ]
  },
  "hooks": {
    "UserPromptSubmit": [
      {
        "type": "command",
        "command": "bash 프로젝트루트/.claude/hooks/prompt-analyzer.sh",
        "timeout": 5000
      },
      {
        "type": "command",
        "command": "bash 프로젝트루트/.claude/hooks/worklog-writer.sh",
        "timeout": 3000
      }
    ],
    "PostToolUse": [
      {
        "type": "command",
        "command": "bash 프로젝트루트/.claude/hooks/code-review-reminder.sh",
        "timeout": 5000
      }
    ]
  }
}
```

### 훅 이벤트 종류

| 이벤트 | 발생 시점 | 이 프로젝트에서의 용도 |
|--------|----------|---------------------|
| `UserPromptSubmit` | 사용자가 프롬프트를 입력했을 때 | 에이전트/챕터 추천 + 작업일지 기록 |
| `PostToolUse` | Claude가 도구(Write/Edit/Bash 등)를 사용한 후 | 코드 리뷰 귀띔 + 문서화 리마인더 |

### 훅 1: prompt-analyzer.sh (선배의 귀띔)

**목적**: 사용자 프롬프트를 분석해서 적절한 에이전트와 스킬북 챕터를 추천한다.

**동작 원리**:
1. stdin에서 JSON을 읽음 (`{"prompt": "사용자 입력 내용"}`)
2. 프롬프트 텍스트를 소문자로 변환
3. 6개 분석 함수를 순차 실행
4. 추천 사항이 있으면 stdout으로 출력 (Claude가 이 출력을 읽고 참고)

**6개 분석 함수와 감지 패턴**:

#### (1) detect_domain_keywords -- 도메인 키워드 감지

| 카테고리 | 감지 키워드 | 추천 |
|---------|-----------|------|
| CODE | 코드, 작성, 구현, 드라이버, 함수, 클래스, 모듈, 리팩토링, spi, i2c, uart, gpio, adc, dac, pwm, interrupt, timer, rtos, freertos, driver, firmware, 펌웨어, 임베디드, embedded, esp32, ads1292, compile, 빌드, build, cmake, platformio | senior-code-architect |
| TEST | 테스트, test, 디버그, debug, 에러, error, 버그, bug, 크래시, crash, 오류, 실패, fail, 동작+안, 이상해, 확인해, 검증, verify, 진단, diagnose, 출력값, 결과+확인, 로그, log, 시리얼, serial, 모니터 | test-diagnose-runner |
| PLAN | 계획, 일정, 스케줄, 마일스톤, 로드맵, 보고서, report, 문서, document, 진행+상황, 변경+사항, 결정, 리스크, risk, 예산, budget, 부품+목록, bom, phase, 단계, 검토, review | ekg-project-planner |
| SIGNAL | 필터, filter, fir, iir, bandpass, lowpass, highpass, notch, 신호+처리, fft, 주파수, 샘플링, snr, 노이즈, ecg, ekg, 심전도, qrs, r-peak, 심박, bpm, 전극, electrode | senior-code-architect + 신호처리 모드 |
| AI/ML | 모델+학습, train, 추론, inference, 데이터셋, dataset, 전처리, cnn, lstm, transformer, tensorflow, pytorch, tflite, 딥러닝, 분류, accuracy, loss, epoch | senior-code-architect + ML 모드 |
| HW | 회로, circuit, pcb, schematic, kicad, 거버, 납땜, 저항, 커패시터, op amp, 아날로그, 전원, 레귤레이터, 배터리, 충전 | ekg-project-planner |
| COMM | bluetooth, ble, wifi, mqtt, websocket, http, api, 서버, 백엔드, 클라우드, aws, firebase, 데이터+전송, 통신, protocol | senior-code-architect |
| UI | 프론트엔드, react, flutter, 앱, 모바일, ui, ux, 화면, 그래프+표시, 시각화, oled, lcd | senior-code-architect |

#### (2) detect_action_patterns -- 요청 유형 감지

| 패턴 | 키워드 | 추천 |
|------|--------|------|
| CREATE | 만들어, 작성해, 생성해, 추가해, 구현해, 개발해, 설계해, create, write, add, implement | senior-code-architect |
| MODIFY | 수정해, 고쳐, 변경해, 업데이트, 개선해, 최적화, 리팩토링, fix, modify, improve, optimize | senior-code-architect |
| ANALYZE | 분석해, 설명해, 알려줘, 확인해, 검토해, 비교해, 왜+그런, 어떻게+동작, 원인, 이유 | (범용) |
| TEST | 테스트해, 검증해, 확인해+동작, 실행해+봐, 돌려봐 | test-diagnose-runner |
| PLAN | 계획+세, 일정+잡, 문서+작성, 보고서, 정리해, 요약해 | ekg-project-planner |

#### (3) detect_file_context -- 파일 경로/확장자 감지

프롬프트에서 파일 경로를 추출하여 해당 도메인 추천:
- `.c`, `.cpp`, `.h`, `.ino` -> 임베디드/펌웨어
- `.py` -> Python/ML
- `.js`, `.ts`, `.tsx` -> 프론트엔드
- `.json`, `.yaml` -> 설정/인프라

#### (4) detect_code_patterns -- 코드 스니펫 감지

프롬프트에 코드가 포함된 경우 언어/프레임워크를 감지:
- `#include`, `void`, `typedef`, `esp_err_t` -> C/C++ 임베디드
- `import numpy`, `import torch`, `.fit()`, `.predict()` -> Python ML
- `void setup()`, `Serial.`, `WiFi.` -> Arduino/PlatformIO
- `error:`, `traceback`, `segfault` -> 에러 메시지 -> test-diagnose-runner

#### (5) detect_resume_request -- "이어서" 감지

키워드: 이어서, 계속, 어제+하던, 지난번, 마저, resume, continue, 어디까지, 진행+상황, 하던+거

감지 시 동작:
1. `.claude/worklog/` 에서 가장 최근 작업일지 찾기
2. 해당 일지의 마지막 20줄을 출력
3. Claude가 이 내용을 보고 이전 작업을 이어갈 수 있음

#### (6) recommend_chapters -- 스킬북 챕터 추천

감지된 도메인에 맞는 챕터 경로를 안내:

| 키워드 | 추천 챕터 |
|--------|----------|
| esp32, spi, i2c, freertos, firmware | ch01-esp32-embedded.md |
| ecg, ekg, 필터, qrs, 심박, 신호 | ch02-signal-ekg.md |
| 보안, security, 암호화, 인증 | ch03-security.md |
| 에러+처리, 워치독, 크래시, 복구 | ch04-error-handling.md |
| 모델+학습, tflite, cnn, lstm, 분류 | ch05-ml-ai.md |
| ble, wifi, mqtt, http, 통신 | ch06-communication.md |
| 회로, pcb, schematic, 전원, 배터리 | ch07-hardware.md |
| 테스트, 검증, 벤치마크, 성능+측정 | ch08-testing.md |
| 계획, 로드맵, 마일스톤, 보고서 | ch09-planning.md |
| 코드+스타일, 네이밍, 리뷰, 컨벤션 | ch10-code-style.md |

**출력 형식** (stdout):

```
============================================
  PROMPT ADVISOR (선배의 귀띔)
============================================
[CODE] senior-code-architect 에이전트 추천: 코드 작성/리뷰/수정 작업 감지됨
[ACTION:CREATE] 새로운 코드/기능 생성 요청 -> senior-code-architect 에이전트 활용 권장

[SKILLBOOK] 관련 챕터 -- Read tool로 열어서 참고하세요:
  ch01 -> .claude/skillbook/ch01-esp32-embedded.md
============================================
```

### 훅 2: worklog-writer.sh (작업일지 자동 기록)

**목적**: 매 프롬프트마다 오늘 날짜의 작업일지에 시간과 요청 내용을 기록한다.

**동작 원리**:
1. stdin에서 JSON 읽음 (`{"prompt": "..."}`)
2. 오늘 날짜로 `.claude/worklog/YYYY-MM-DD.md` 파일 결정
3. 파일이 없으면 헤더 생성, 있으면 기존 파일에 추가
4. `### [HH:MM:SS]` + 요청 내용 첫 200자를 기록

**stdin 포맷** (Claude Code가 전달):
```json
{"prompt": "사용자가 입력한 프롬프트 텍스트"}
```

**stdout**: 없음 (파일 기록만 수행)

**생성되는 파일 예시** (`.claude/worklog/2026-03-29.md`):

```markdown
# 작업일지 -- 2026-03-29

## 작업 타임라인

### [14:23:15]
- **요청**: ESP32에서 ADS1292R SPI 드라이버를 작성해줘

### [14:45:02]
- **요청**: 테스트해봐. 내부 테스트 신호가 제대로 나오는지 확인

### [15:10:33]
- **요청**: 필터링 함수 추가해줘. 60Hz 노치 필터
```

### 훅 3: code-review-reminder.sh (코드 리뷰 귀띔)

**목적**: Write/Edit/Bash 도구 사용 후, 코드에서 위험 요소/누락을 감지하여 "이것도 확인했어?" 스타일의 부드러운 리마인더를 제공한다. 강제 차단이 아닌 귀띔이다.

**동작 원리**:
1. stdin에서 JSON 읽음 (`{"tool_name": "Write", "tool_input": {"file_path": "...", "content": "..."}}`)
2. `tool_name`이 Write/Edit/Bash가 아니면 즉시 종료
3. 파일 확장자와 내용을 분석
4. 6개 체크 함수를 순차 실행
5. 발견 사항을 JSON으로 stdout에 출력

**stdin 포맷**:
```json
{
  "tool_name": "Write",
  "tool_input": {
    "file_path": "src/driver/ads1292r.c",
    "content": "작성된 코드 내용..."
  }
}
```

**stdout 포맷** (발견 사항 있을 때):
```json
{
  "hookSpecificOutput": {
    "additionalContext": "[선배의 코드 리뷰 귀띔] 이것도 확인했어?\n  - malloc 사용했는데 NULL 체크 추가했어?\n참고로 강제 사항 아니야. 이미 확인했으면 넘어가도 돼!"
  }
}
```

**6개 체크 함수와 감지 패턴**:

#### (1) check_dangerous_operations -- 위험한 작업

| 감지 패턴 | 귀띔 내용 |
|----------|----------|
| `malloc`/`calloc`/`realloc` + NULL 체크 없음 | "NULL 체크 추가했어? 메모리 할당 실패 시 크래시" |
| `strcpy`/`sprintf`/`gets` | "안전하지 않은 함수. strncpy/snprintf/fgets로 교체" |
| 고정 크기 버퍼 `[128]`, `[256]` 등 | "입력 데이터가 버퍼보다 클 수 있는지 확인" |
| `while(true)` + break/delay 없음 | "워치독 타임아웃 위험. ESP32에서 특히 조심" |
| `fopen`/`socket` + close 없음 | "리소스 릭 확인했어?" |
| `rm -rf`/`DROP TABLE`/`nvs_flash_erase` | "파괴적 작업. 진짜 의도한 거 맞아?" |
| 긴 delay (4자리 이상) | "실시간 처리에 영향 없는지 확인. 태스크로 분리는?" |

#### (2) check_error_handling -- 에러 처리

| 감지 패턴 | 귀띔 내용 |
|----------|----------|
| ESP-IDF 함수 호출 + 반환값 미체크 | "ESP_ERROR_CHECK나 if 분기 확인" |
| Python 파일/네트워크 코드 + try-except 없음 | "예외 처리 추가했어?" |
| 나눗셈 + 분모 검증 없음 (sample_count 등) | "분모가 0일 수 있는 경우 체크" |
| ISR/callback/IRAM_ATTR | "긴 작업이나 printf 쓰면 안 됨. IRAM_ATTR 붙였어?" |
| 포인터 역참조 `->` + NULL 체크 없음 | "포인터가 NULL일 수 있는 경우 체크" |

#### (3) check_security -- 보안

| 감지 패턴 | 귀띔 내용 |
|----------|----------|
| `password=`, `api_key=`, `token=` 하드코딩 | "설정 파일/환경변수로 빼는 게 좋지 않을까?" |
| SQL 쿼리 + 사용자 입력 직접 삽입 | "parameterized query 쓰는 거 알지?" |
| `recv`/`read`/`input` + 검증 없음 | "입력값 검증/범위 체크 추가했어?" |
| 디버그 코드 잔존 (`debug`, `TODO`, `FIXME`, `HACK`) | "프로덕션 전에 정리할 거 맞지?" |
| 평문 통신 (`http://`, `mqtt://`, `ftp://`) | "민감한 데이터 있으면 암호화 고려" |
| `chmod 777` | "최소 권한 원칙 기억하지?" |

#### (4) check_embedded_specific -- ESP32/임베디드 특화

| 감지 패턴 | 귀띔 내용 |
|----------|----------|
| `xTaskCreate` + 스택 512~2048 | "스택 작은 편. 큰 로컬 변수/재귀 시 오버플로우 위험" |
| SPI/I2C 접근 + mutex 없음 | "멀티태스크 환경이면 뮤텍스로 보호" |
| ADC/센서 읽기 + 필터링 없음 | "노이즈 줄이려면 평균/필터링 적용은?" |
| sleep 모드 사용 | "깨어난 후 주변기기 재초기화 확인" |
| DMA 버퍼 + alignment 없음 | "ESP32 DMA는 4바이트 정렬 필요" |

#### (5) check_bash_commands -- Bash 명령어

| 감지 패턴 | 귀띔 내용 |
|----------|----------|
| `rm -rf /`, `dd if=`, `mkfs` | "파괴적 명령어. 경로 한 번 더 확인" |
| `sudo`, `chmod`, `chown` | "대상 범위가 의도한 것과 맞는지 확인" |
| `git push --force`, `git reset --hard` | "되돌리기 어려움. 정말 필요한 건지 확인" |

#### (6) check_documentation_needs -- 맥락노트/체크리스트 리마인더

Write/Edit으로 **의미 있는 변경**(새 파일 생성, 새 함수/클래스/모듈, 아키텍처 결정)을 했을 때:

```
[맥락노트 리마인더]
  방금 의미 있는 변경을 했어. 다음을 기록해두면 나중에 "왜 이렇게 했지?" 안 헤매:
  - 맥락노트 작성: docs/context-notes/CN-XXXX-제목.md
  - 체크리스트 업데이트: docs/checklist.md
```

단, `docs/` 하위 파일이나 `.md` 파일 수정은 문서 작업 자체이므로 리마인더를 띄우지 않는다.

### 새 프로젝트에서 훅 커스텀하기

**경로 변경**: 스크립트 내부의 `PROJECT_ROOT` 변수를 새 프로젝트 경로로 수정:

```bash
# 변경 전
PROJECT_ROOT="G:/AIEKG"

# 변경 후
PROJECT_ROOT="/path/to/your/project"
```

**키워드 추가**: 각 감지 함수의 `grep -qiE` 패턴에 키워드를 추가한다. 패턴은 정규식이므로 `|`로 구분:

```bash
# 예: React 프로젝트에 맞게 UI 키워드 추가
if echo "$prompt_lower" | grep -qiE "컴포넌트|component|useState|useEffect|redux|zustand|tailwind"; then
    RECOMMENDATIONS+="[UI] React 컴포넌트 작업 감지\n"
fi
```

---

## 6. 스킬북 챕터 시스템

### 왜 챕터로 분리했나?

**대안 비교**:

| 방식 | 문제 |
|------|------|
| 모든 가이드를 에이전트 파일에 넣기 | 토큰 낭비. ESP32 작업 시 ML 가이드까지 로드 |
| 에이전트마다 자기 도메인 매뉴얼 내장 | 중복. ch01(ESP32)은 코드 에이전트와 테스트 에이전트 모두 필요 |
| **챕터로 분리 (채택)** | 중복 제거, on-demand 로딩, 챕터 단위 유지보수 |

### 에이전트가 어떻게 참조하는가?

1. **자동 추천 경로**: prompt-analyzer 훅이 프롬프트를 분석 -> 관련 챕터 번호 출력 -> Claude가 Read tool로 해당 챕터를 열어서 참고
2. **직접 참조 경로**: 에이전트 body에 있는 Skillbook 테이블을 보고, 에이전트가 스스로 판단하여 필요한 챕터를 Read
3. **복합 참조**: 여러 챕터를 동시에 열 수 있음 (예: ch01 + ch02 = ESP32에서 EKG 신호처리)

### 챕터 목록과 용도

| Ch | 파일명 | 주제 | 언제 사용? |
|----|--------|------|-----------|
| 01 | ch01-esp32-embedded.md | ESP32 임베디드 | SPI/I2C/GPIO, FreeRTOS, 메모리, 전원관리, OTA |
| 02 | ch02-signal-ekg.md | 신호처리/EKG | ADC, 필터(FIR/IIR/Notch), QRS 검출, 노이즈 제거 |
| 03 | ch03-security.md | 보안 | 자격증명, 입력검증, 통신암호화, OWASP 임베디드 |
| 04 | ch04-error-handling.md | 에러 처리 | ESP-IDF 에러, 센서 실패 복구, 워치독, 로깅 |
| 05 | ch05-ml-ai.md | AI/ML | ECG 분류, TFLite 추론, 데이터 전처리, 모델 경량화 |
| 06 | ch06-communication.md | 통신 | BLE, WiFi, MQTT, HTTP, 패킷 설계, 재전송 |
| 07 | ch07-hardware.md | 하드웨어 | ADS1292R 스펙, 아날로그 프론트엔드, PCB, BOM |
| 08 | ch08-testing.md | 테스트 | 단위/통합/시스템, 신호 품질, 벤치마크 기준 |
| 09 | ch09-planning.md | 기획 | 7Phase 로드맵, 문서 템플릿, 의사결정, 리스크 |
| 10 | ch10-code-style.md | 코드 스타일 | 네이밍, SOLID, 변경기록, 리뷰 체크리스트 |

### _index.md 파일 구조

```markdown
# 프로젝트 Skillbook -- 목차

> 필요한 챕터만 Read tool로 열어서 참고하세요.
> 경로: `.claude/skillbook/chXX-이름.md`

| Ch | 파일 | 주제 | 언제 펼쳐볼까? |
|----|------|------|----------------|
| 01 | `ch01-esp32-embedded.md` | ESP32 임베디드 개발 | SPI/I2C/GPIO, FreeRTOS... |
| ... | ... | ... | ... |

## 사용법

1. **훅이 자동 추천** -- prompt-analyzer가 관련 챕터 번호를 알려줌
2. **직접 참조** -- 에이전트가 필요한 챕터를 Read tool로 확인
3. **복합 작업** -- 여러 챕터를 동시에 참조 가능
```

### 새 챕터 추가 방법

1. `.claude/skillbook/ch11-새주제.md` 파일 생성
2. `_index.md` 테이블에 행 추가
3. 관련 에이전트의 Skillbook 테이블에 행 추가
4. `prompt-analyzer.sh`의 `recommend_chapters()` 함수에 감지 패턴 추가:

```bash
# ch11 추가 예시
if echo "$prompt_lower" | grep -qiE "새키워드1|새키워드2|새키워드3"; then
    CHAPTERS+="  ch11 -> $SKILLBOOK/ch11-새주제.md\n"
fi
```

---

## 7. 3종 문서 체계

### 전체 구조

```
               기획서 (설계도)
              "뭘 만들 건가?"
              docs/plan/
                  |
                  |  결정이 내려질 때마다
                  v
            맥락노트 (시방서)
           "왜 이렇게 했나?"
          docs/context-notes/
                  |
                  |  작업이 완료될 때마다
                  v
           체크리스트 (공정표)
          "어디까지 했나?"
          docs/checklist.md
```

### 기획서 (설계도) -- docs/plan/

**역할**: 프로젝트의 전체 로드맵, 목표, 범위, 일정을 정의한다. "무엇을 만들 것인가?"에 대한 답.

**작성 시점**: 프로젝트 시작 시 (Phase 0). 큰 방향 전환 시 버전 업데이트.

**템플릿**:

```markdown
# 프로젝트명 v1.0

## 1. 프로젝트 개요
- 목표: (한 문장으로)
- 핵심 기술: (주요 기술 스택)
- 제약 조건: (예산, 기간, 인원 등)

## 2. 시스템 아키텍처
(전체 시스템 구성도)

## 3. 개발 로드맵
### Phase 1: ...
### Phase 2: ...
(각 Phase의 목표, 기간, Gate 판정 기준)

## 4. BOM (부품 목록)
| 부품 | 수량 | 예상 비용 |

## 5. 리스크 관리
| 리스크 | 확률 | 영향 | 대응 |
```

### 맥락노트 (시방서) -- docs/context-notes/

**역할**: 의미 있는 의사결정의 근거를 기록한다. "왜 이렇게 했는가?"에 대한 답. 6개월 후의 나, 또는 이 프로젝트에 처음 합류한 사람이 읽어도 이해할 수 있어야 한다.

**작성 시점**: 의미 있는 결정을 내렸을 때. "모든 변경마다" 쓰면 피로하므로 선별적으로.

- code-review-reminder 훅이 의미 있는 변경(새 파일, 새 모듈, 아키텍처 결정) 후 자동으로 리마인더를 띄운다.

**파일 명명 규칙**: `CN-XXXX-간단한제목.md` (XXXX = 순번)

**_index.md 구조**:

```markdown
# 맥락노트 (Context Notes) -- 시방서

> "왜 이렇게 결정했는가?"를 기록하는 곳.

## 작성 원칙
1. **결정 사항**을 먼저 쓴다 -- "X를 Y로 했다"
2. **배경/맥락**을 쓴다 -- 그때 상황이 어땠는지
3. **검토한 대안**을 쓴다 -- 다른 선택지는 뭐가 있었고 왜 기각했는지
4. **근거**를 쓴다 -- 왜 이 결정이 최선이었는지
5. **영향/주의사항**을 쓴다 -- 이 결정 때문에 나중에 주의할 점

## 파일 명명 규칙
`CN-XXXX-간단한제목.md` (XXXX = 순번)

## 노트 목록
(최신순)
- [CN-0001](CN-0001-dev-infra-design.md) -- 개발 인프라 설계 결정 근거
```

**맥락노트 템플릿**:

```markdown
# CN-XXXX: 제목

- 작성일: YYYY-MM-DD
- 단계: Phase N (단계명)

## 결정 사항
(무엇을 어떻게 결정했는지)

## 배경/맥락
(그때 상황이 어떠했는지)

## 검토한 대안
### 대안 1: ...
- **기각 사유**: ...

### 대안 2: ...
- **기각 사유**: ...

### 대안 3 (채택): ...
- **채택 사유**: ...

## 근거
(왜 이 방식이 최선이었는지)

## 주의사항
(이 결정의 제한사항, 향후 주의할 점)
```

### 체크리스트 (공정표) -- docs/checklist.md

**역할**: Phase별 진행 상황을 추적한다. "어디까지 했는가?"에 대한 답.

**작성 시점**: Phase 0에서 전체 골격 생성. 작업 완료 시마다 업데이트.

**상태 아이콘**:

| 아이콘 | 의미 |
|--------|------|
| `- ✅` | 완료 |
| `- 🔄` | 진행중 |
| `- ⬚` | 미착수 |
| `- ⏸️` | 보류 |
| `- ❌` | 기각 |

**구조**:

```markdown
# 프로젝트 체크리스트 (공정표)

> 기획서(설계도)를 따라가며 진행 상황을 추적.
> ✅ 완료 / 🔄 진행중 / ⬚ 미착수 / ⏸️ 보류 / ❌ 기각

---
마지막 업데이트: YYYY-MM-DD

## Phase 0: 개발 환경 구축
- ✅ 개발환경 선정
- ✅ 에이전트 시스템 구축
  - ✅ 에이전트 A
  - ✅ 에이전트 B
- ⬚ 기획서 정식 작성

## Phase 1: 핵심 기능 A (Gate 1)
> 목표: (Gate 통과 기준)

- ⬚ 세부 작업 1
- ⬚ 세부 작업 2
- ⬚ **Gate 1 판정**: 성공 ✅/❌

---

## 다음 할 일 (Next Actions)
1. ...
2. ...

---

## 변경 이력
| 날짜 | 변경 내용 |
|------|-----------|
| YYYY-MM-DD | 초기 생성 |
```

### 3종 문서의 상호 관계

```
사용자가 "ADS1292R 대신 ADS1293 쓰자"고 결정
       |
       v
[1] 기획서 수정 -- BOM에서 칩 변경, Phase 1 일정 영향 반영
       |
       v
[2] 맥락노트 작성 -- CN-0005-adc-chip-change.md
    "ADS1293으로 변경. 이유: 더 낮은 전력, 3채널, 비용 차이 미미.
     대안: ADS1292R(기각: 단종 예정), MAX30003(기각: 비쌈).
     주의: SPI 프로토콜이 다름, 드라이버 재작성 필요."
       |
       v
[3] 체크리스트 업데이트 -- "SPI 드라이버 구현" 항목을 "ADS1293용 SPI 드라이버"로 수정
```

---

## 8. 작업일지 시스템

### 자동 기록 방식

1. 사용자가 프롬프트를 입력할 때마다 `UserPromptSubmit` 훅이 발동
2. `worklog-writer.sh`가 실행되어 다음을 수행:
   - 오늘 날짜(`YYYY-MM-DD`)로 파일명 결정
   - `.claude/worklog/` 디렉토리에 파일이 없으면 헤더 생성
   - 현재 시각(`HH:MM:SS`)과 프롬프트 내용(첫 200자)을 추가 기록
3. 결과: 하루의 작업이 시간순으로 자동 누적됨

### "어제 하던거 이어서" 동작 원리

사용자가 "이어서", "어제 하던거", "계속" 같은 키워드를 입력하면:

1. `prompt-analyzer.sh`의 `detect_resume_request()` 함수가 감지
2. `.claude/worklog/` 디렉토리에서 가장 최근 날짜의 `.md` 파일을 찾음 (`ls -t | head -1`)
3. 해당 파일의 마지막 20줄을 읽어서 stdout에 출력
4. Claude가 이 내용을 보고 이전 작업 맥락을 파악

**예시 시나리오**:

```
[월요일에 SPI 드라이버 작업 중 퇴근]

.claude/worklog/2026-03-28.md:
### [17:45:12]
- **요청**: SPI 드라이버에서 DRDY 인터럽트 핸들러 추가해줘

[화요일 출근 후]
사용자: "어제 하던거 이어서 해줘"

prompt-analyzer 출력:
[RESUME] 이전 작업 이어하기 요청 감지!
[RESUME] 최근 작업일지 발견: 2026-03-28
[RESUME] 작업일지 경로: .claude/worklog/2026-03-28.md
[RESUME] === 최근 작업 내용 ===
### [17:45:12]
- **요청**: SPI 드라이버에서 DRDY 인터럽트 핸들러 추가해줘
[RESUME] === 끝 ===

-> Claude가 이 맥락을 보고 DRDY 인터럽트 핸들러 작업을 이어감
```

### 일지 파일 관리

- 날짜별로 자동 분리되므로 별도 관리 불필요
- 오래된 일지는 삭제해도 시스템 동작에 영향 없음 (최근 일지만 참조)
- 일지는 `.claude/worklog/`에 저장되므로 `.gitignore`에 추가 가능

---

## 9. 새 프로젝트에 적용하는 방법

### Step 1: 디렉토리 구조 생성

```bash
# 프로젝트 루트에서 실행
PROJECT_ROOT=$(pwd)

# .claude 기본 구조
mkdir -p .claude/agents
mkdir -p .claude/hooks
mkdir -p .claude/skillbook
mkdir -p .claude/worklog
mkdir -p .claude/agent-memory

# docs 구조
mkdir -p docs/plan
mkdir -p docs/context-notes
```

### Step 2: 훅 스크립트 복사 + 경로 수정

원본 프로젝트에서 3개 훅 스크립트를 복사한다:

```bash
# 복사
cp 원본/.claude/hooks/prompt-analyzer.sh  .claude/hooks/
cp 원본/.claude/hooks/worklog-writer.sh   .claude/hooks/
cp 원본/.claude/hooks/code-review-reminder.sh .claude/hooks/
```

각 스크립트에서 `PROJECT_ROOT`를 새 프로젝트 경로로 수정한다:

```bash
# prompt-analyzer.sh, worklog-writer.sh 에서:
PROJECT_ROOT="새/프로젝트/경로"
```

### Step 3: settings.local.json 작성

`.claude/settings.local.json`을 생성한다. 경로를 새 프로젝트에 맞게 수정:

```json
{
  "permissions": {
    "allow": [
      "Bash(새/프로젝트/경로/.claude/hooks/*)",
      "Bash(bash \"새/프로젝트/경로/.claude/hooks/prompt-analyzer.sh\")",
      "Bash(bash \"새/프로젝트/경로/.claude/hooks/code-review-reminder.sh\")"
    ]
  },
  "hooks": {
    "UserPromptSubmit": [
      {
        "type": "command",
        "command": "bash 새/프로젝트/경로/.claude/hooks/prompt-analyzer.sh",
        "timeout": 5000
      },
      {
        "type": "command",
        "command": "bash 새/프로젝트/경로/.claude/hooks/worklog-writer.sh",
        "timeout": 3000
      }
    ],
    "PostToolUse": [
      {
        "type": "command",
        "command": "bash 새/프로젝트/경로/.claude/hooks/code-review-reminder.sh",
        "timeout": 5000
      }
    ]
  }
}
```

### Step 4: 에이전트 정의

새 프로젝트에 맞는 에이전트를 `.claude/agents/`에 작성한다.

**범용 템플릿** (프로젝트에 맞게 수정):

```markdown
---
name: 에이전트-이름
description: "이 에이전트를 언제 사용해야 하는지에 대한 상세 설명.
Examples와 함께 작성하면 Claude가 더 정확하게 판단한다."
model: opus
color: blue
memory: project
---

You are a [역할 설명]. [핵심 원칙].

## Core Responsibilities
1. **역할 1** -- 설명
2. **역할 2** -- 설명
3. **역할 3** -- 설명

## Skillbook (상세 가이드)
| 상황 | 챕터 파일 |
|------|-----------|
| ... | `.claude/skillbook/ch01-xxx.md` |

## 출력 형식
(에이전트가 반드시 따를 보고 형식)

## Quality Self-Check
- [ ] 체크 항목 1
- [ ] 체크 항목 2
```

**에이전트 메모리 디렉토리**: 에이전트마다 `.claude/agent-memory/에이전트이름/` 디렉토리를 생성한다.

```bash
mkdir -p .claude/agent-memory/에이전트이름
```

### Step 5: 스킬북 챕터 작성

프로젝트 도메인에 맞는 챕터를 `.claude/skillbook/`에 작성한다.

1. `_index.md` (목차) 생성
2. 도메인별 챕터 파일 생성 (`ch01-xxx.md`, `ch02-yyy.md`, ...)

챕터 내용은 해당 도메인의 상세 가이드, 코딩 규칙, 참조 자료를 담는다.

### Step 6: prompt-analyzer 키워드 커스텀

`prompt-analyzer.sh`에서 프로젝트 도메인에 맞게 키워드를 수정한다.

**수정할 함수들**:

1. `detect_domain_keywords()` -- 도메인 키워드를 새 프로젝트에 맞게 교체
2. `detect_action_patterns()` -- 보통 범용이므로 수정 불필요
3. `detect_file_context()` -- 프로젝트 파일 확장자에 맞게 조정
4. `detect_code_patterns()` -- 사용하는 언어/프레임워크에 맞게 수정
5. `recommend_chapters()` -- 새 스킬북 챕터에 맞게 전체 교체

### Step 7: code-review-reminder 커스텀

`code-review-reminder.sh`에서 프로젝트 기술 스택에 맞게 수정한다.

- C/C++ 임베디드가 아닌 프로젝트라면 `check_embedded_specific()`을 제거하거나 다른 체크로 교체
- Python 웹 프로젝트라면 Django/FastAPI 특화 체크 추가
- 보안 체크(`check_security()`)는 대부분 범용이므로 그대로 유지

### Step 8: 문서 체계 초기화

```bash
# 맥락노트 인덱스
cat > docs/context-notes/_index.md << 'EOF'
# 맥락노트 (Context Notes) -- 시방서

> "왜 이렇게 결정했는가?"를 기록하는 곳.

## 작성 원칙
1. **결정 사항**을 먼저 쓴다
2. **배경/맥락**을 쓴다
3. **검토한 대안**을 쓴다
4. **근거**를 쓴다
5. **영향/주의사항**을 쓴다

## 파일 명명 규칙
`CN-XXXX-간단한제목.md` (XXXX = 순번)

## 노트 목록
(최신순)
EOF

# 체크리스트 초기화
cat > docs/checklist.md << 'EOF'
# 프로젝트 체크리스트 (공정표)

> ✅ 완료 / 🔄 진행중 / ⬚ 미착수 / ⏸️ 보류 / ❌ 기각

---
마지막 업데이트: YYYY-MM-DD

## Phase 0: 개발 환경 구축
- ⬚ 개발환경 선정
- ⬚ 에이전트 시스템 구축
- ⬚ 훅 시스템 구축
- ⬚ 스킬북 작성
- ⬚ 문서 체계 초기화

---

## 다음 할 일 (Next Actions)
1. ...

---

## 변경 이력
| 날짜 | 변경 내용 |
|------|-----------|
EOF
```

### Step 9: 검증

모든 설정이 끝나면 Claude Code를 실행하여 다음을 확인한다:

1. **훅 동작 확인**: 아무 프롬프트나 입력하면 prompt-analyzer의 추천이 보이는가?
2. **작업일지 확인**: `.claude/worklog/오늘날짜.md` 파일이 생성되었는가?
3. **에이전트 확인**: Agent tool로 에이전트를 호출할 수 있는가?
4. **코드 리뷰 확인**: Write/Edit 사용 후 귀띔이 뜨는가?

---

## 10. 커스터마이징 가이드

### 에이전트 추가

1. `.claude/agents/새에이전트.md` 파일 생성 (frontmatter + body)
2. `.claude/agent-memory/새에이전트/` 디렉토리 생성
3. `prompt-analyzer.sh`의 `detect_domain_keywords()`에 새 에이전트 추천 패턴 추가:

```bash
if echo "$prompt_lower" | grep -qiE "새도메인키워드1|새도메인키워드2"; then
    RECOMMENDATIONS+="[NEW] 새에이전트 에이전트 추천: 새 도메인 작업 감지됨\n"
fi
```

4. 관련 스킬북 챕터가 있다면 에이전트 body의 Skillbook 테이블에 추가

### 훅 패턴 추가

**prompt-analyzer에 감지 패턴 추가**:

각 `detect_*()` 함수 내에서 `grep -qiE` 뒤의 패턴 문자열에 `|새키워드`를 추가하면 된다.

```bash
# 기존
if echo "$prompt_lower" | grep -qiE "코드|작성|구현"; then

# 키워드 추가 후
if echo "$prompt_lower" | grep -qiE "코드|작성|구현|새로운키워드"; then
```

**code-review-reminder에 체크 패턴 추가**:

새로운 체크 함수를 만들고, 실행부에 추가한다:

```bash
# 새 체크 함수 정의
check_my_custom_pattern() {
    if echo "$CONTENT" | grep -qE '위험한패턴'; then
        REMINDERS+="  - 위험한 패턴 발견. 이것 확인했어?\n"
    fi
}

# 실행부에 추가
check_dangerous_operations
check_error_handling
check_security
check_embedded_specific
check_bash_commands
check_my_custom_pattern      # <-- 추가
check_documentation_needs
```

### 챕터 추가

1. `.claude/skillbook/chNN-새주제.md` 파일 생성
2. `.claude/skillbook/_index.md` 테이블에 행 추가
3. 관련 에이전트의 Skillbook 테이블에 행 추가
4. `prompt-analyzer.sh`의 `recommend_chapters()`에 패턴 추가:

```bash
# 새 챕터 추천
if echo "$prompt_lower" | grep -qiE "키워드1|키워드2"; then
    CHAPTERS+="  chNN -> $SKILLBOOK/chNN-새주제.md\n"
fi
```

### 키워드 수정 원칙

1. **한국어 + 영어 병행**: 사용자가 어느 언어로든 입력할 수 있으므로 양쪽 다 등록
2. **복합 키워드는 `.*`로**: "신호 처리" -> `신호.*처리` (중간에 뭐가 와도 매칭)
3. **단어 경계 주의**: `\b`는 한국어에서 동작 안 함. 한국어 키워드는 그냥 부분 매칭
4. **너무 짧은 키워드 피하기**: "앱"처럼 짧으면 오탐이 많음. "앱.*개발"처럼 구체화
5. **테스트**: 패턴 수정 후 실제 프롬프트로 테스트하여 오탐/미탐 확인

### 에이전트 description 작성 팁

description은 Claude가 "이 에이전트를 언제 쓸지" 판단하는 핵심 텍스트다.

**좋은 description**:
- Examples를 3개 이상 포함 (User -> Assistant 형식)
- 발동 조건뿐 아니라 **발동하지 않아야 할 경우**도 명시
- 사용자가 한국어/영어 섞어 쓸 수 있으므로 예시도 혼합

**나쁜 description**:
- "코드 관련 작업에 사용" -- 너무 추상적
- 예시 없음 -- Claude가 구체적 상황을 판단하기 어려움

---

## 부록: 전체 설정 체크리스트

새 프로젝트에 이 시스템을 적용할 때 빠뜨림 없이 확인하기 위한 체크리스트:

```
[ ] .claude/settings.local.json -- 훅 등록, 권한, 경로 설정
[ ] .claude/hooks/prompt-analyzer.sh -- 복사 + PROJECT_ROOT 수정 + 키워드 커스텀
[ ] .claude/hooks/worklog-writer.sh -- 복사 + PROJECT_ROOT 수정
[ ] .claude/hooks/code-review-reminder.sh -- 복사 + 기술 스택에 맞게 체크 함수 수정
[ ] .claude/agents/ -- 프로젝트에 맞는 에이전트 정의 (최소 1개)
[ ] .claude/agent-memory/ -- 에이전트별 메모리 디렉토리 생성
[ ] .claude/skillbook/_index.md -- 챕터 목차
[ ] .claude/skillbook/ch*.md -- 도메인별 챕터 (최소 1개)
[ ] docs/plan/ -- 기획서 디렉토리 (내용은 프로젝트 시작 시 작성)
[ ] docs/context-notes/_index.md -- 맥락노트 인덱스
[ ] docs/checklist.md -- 체크리스트 초기화
[ ] 훅 동작 검증 -- 프롬프트 입력 시 추천 출력 확인
[ ] 작업일지 생성 확인 -- .claude/worklog/ 파일 생성 확인
[ ] 코드 리뷰 훅 확인 -- Write/Edit 후 귀띔 출력 확인
```
