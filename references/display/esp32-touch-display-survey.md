# ESP32-WROOM-32D 터치 디스플레이 조사 보고서

- 작성일: 2026-04-02
- 작성자: reference-researcher
- 목적: 병실 베드사이드 EKG 장치용 실시간 심전도 파형 표시 디스플레이 선정

---

## 1. SPI 5인치(800x480) 디스플레이의 Refresh Rate 한계

### 이론적 대역폭 계산

```
해상도: 800 x 480 = 384,000 픽셀
색심도: 16비트(RGB565) = 2 bytes/pixel
1프레임 데이터량: 384,000 x 2 = 768,000 bytes = 750 KB

ESP32 SPI 최대 클록: 80 MHz (APB_CLK/1, SPI_DEVICE_NO_DUMMY 사용 시)
실제 안정 클록: 40 MHz (대부분의 디스플레이 컨트롤러 한계)

@ 80 MHz SPI: 80,000,000 / 8 = 10 MB/s -> 750KB / 10MB = 75ms -> ~13 FPS (이론상)
@ 40 MHz SPI: 40,000,000 / 8 = 5 MB/s  -> 750KB / 5MB  = 150ms -> ~6.6 FPS (이론상)
@ 20 MHz SPI: 20,000,000 / 8 = 2.5 MB/s -> 750KB / 2.5MB = 300ms -> ~3.3 FPS
```

### 결론: 5인치 800x480 SPI 전체화면 갱신은 매우 느림

- **전체 화면 갱신 기준 6~13 FPS** -- 심전도 파형 표시에는 최소 25~30 FPS 필요
- 단, **부분 갱신(Partial Update)** 기법 사용 시 실용적으로 가능
  - 파형 영역만 갱신: 800 x 200 픽셀 = 312 KB -> 약 15~32 FPS 가능
  - TFT_eSPI Sprite + scroll 기법으로 flicker-free 구현 가능

---

## 2. 드라이버 IC별 비교 분석

| 항목 | RA8875 | SSD1963 | ST7796 | ILI9488 | ILI9806 |
|------|--------|---------|--------|---------|---------|
| **해상도** | 최대 800x480 | 최대 864x480 | 최대 480x320 | 최대 480x320 | 최대 480x800 |
| **인터페이스** | SPI/I2C/Parallel | Parallel 8/16비트만 | SPI/Parallel | SPI/Parallel | Parallel만 |
| **SPI 최대 클록** | ~35 MHz (실측) | N/A (SPI 미지원) | 80 MHz (ESP32 실측) | 20 MHz | N/A |
| **HW 가속** | 선/사각/원/타원 그리기 | 없음 | 없음 | 없음 | 없음 |
| **TFT_eSPI** | 비공식 (포크 존재) | 지원 (Parallel만) | 공식 지원 | 공식 지원 | 미지원 |
| **LovyanGFX** | Issue 단계 | 미확인 | 공식 지원 | 공식 지원 | 미확인 |
| **MISO tristate** | 정상 | N/A | 일부 보드 문제 | 미지원 (버스 충돌) | N/A |
| **5인치 가용** | O (800x480) | O (800x480) | X (최대 4인치) | X (최대 3.5인치) | O (미지원 IC) |

### 핵심 발견사항

1. **ILI9806**: ESP32/Arduino 생태계에서 거의 지원되지 않음. STM32 위주. 사실상 선택지에서 제외.
2. **SSD1963**: SPI 미지원, Parallel 8비트만 가능. ESP32에서 GPIO 8개+제어 3~4개 = 총 11~12핀 소모. 가능은 하지만 핀 부족 위험.
3. **RA8875**: SPI 지원 + 800x480 + HW 가속. 5인치 SPI 디스플레이의 사실상 유일한 선택지. 단, TFT_eSPI 공식 미지원 (Adafruit_RA8875 라이브러리 또는 LovyanGFX 포크 사용).
4. **ST7796**: ESP32 생태계 최적화 (80MHz, TFT_eSPI 공식 지원). 단, 최대 4인치 480x320.
5. **ILI9488**: SPI MISO가 tristate 안 됨 -> ADS1292R과 SPI 버스 공유 불가 (치명적 문제). 제외.

---

## 3. SPI 버스 공유 분석 (ADS1292R + 디스플레이)

### ESP32 SPI 버스 구조
```
ESP32 SPI 버스:
- SPI0, SPI1: 내부 Flash 전용 (사용 금지)
- SPI2 (HSPI): 사용 가능
- SPI3 (VSPI): 사용 가능
```

### 방안 1: 동일 SPI 버스 + 별도 CS (권장하지 않음)
- ADS1292R: CPOL=0, CPHA=1 (SPI Mode 1), 최대 4 MHz
- 디스플레이: 드라이버에 따라 SPI Mode 0 또는 3, 40~80 MHz
- SPI 모드 불일치 + 클록 속도 차이 -> 동적 전환 필요 -> 타이밍 리스크

### 방안 2: 별도 SPI 버스 사용 (강력 권장)
```
HSPI (SPI2) -> ADS1292R (저속, Mode 1, 전용)
  - SCLK: GPIO 14
  - MOSI: GPIO 13
  - MISO: GPIO 12
  - CS:   GPIO 15

VSPI (SPI3) -> 디스플레이 (고속, 전용)
  - SCLK: GPIO 18
  - MOSI: GPIO 23
  - MISO: GPIO 19 (RA8875 읽기 시) 또는 미연결 (ST7796)
  - CS:   GPIO 5
```

### 방안 3: 디스플레이 MISO 미연결 (ST7796/ILI9341 등)
- 대부분의 경우 디스플레이에서 데이터를 읽을 필요 없음
- MISO를 연결하지 않으면 SPI 버스 충돌 원천 차단
- TFT_eSPI에서 `TFT_MISO -1` 설정으로 비활성화 가능

### 핀 할당 요약 (방안 2 기준)

| 기능 | GPIO | 비고 |
|------|------|------|
| ADS1292R SCLK | 14 | HSPI |
| ADS1292R MOSI | 13 | HSPI |
| ADS1292R MISO | 12 | HSPI |
| ADS1292R CS | 15 | |
| ADS1292R DRDY | 4 | 인터럽트 |
| ADS1292R START | 2 | |
| ADS1292R RESET | 27 | |
| Display SCLK | 18 | VSPI |
| Display MOSI | 23 | VSPI |
| Display MISO | 19 | VSPI (선택) |
| Display CS | 5 | |
| Display DC/RS | 21 | 커맨드/데이터 |
| Display RST | 22 | |
| Display BL | 25 | 백라이트 PWM |
| Touch CS | 26 | (SPI 터치 시) |
| Wi-Fi | 내장 | 별도 핀 불필요 |

**총 사용 핀: 15개** -- ESP32-WROOM-32D의 가용 GPIO로 충분히 커버 가능.

---

## 4. 터치 컨트롤러 비교

| 항목 | 저항식 (XPT2046) | 정전식 (GT911/FT5x06) |
|------|------------------|----------------------|
| 인터페이스 | SPI (추가 CS 핀 필요) | I2C (SDA/SCL 공유) |
| 장갑 사용 | 가능 | 불가 (일부 특수 장갑 제외) |
| 캘리브레이션 | 필요 (3점 터치) | 불필요 (컨트롤러 내장) |
| 멀티터치 | 불가 | 가능 (최대 5점) |
| 내구성 | 표면 마모 우려 | 표면 강화유리, 내구성 우수 |
| 의료 환경 적합성 | 장갑 사용 가능 -> 유리 | 직관적 조작 -> 유리 |
| 추가 핀 소모 | SPI CS 1개 (GPIO 26) | I2C 2개 (SDA/SCL) |
| 가격 | 저렴 | 약간 비쌈 |

### 의료 환경 권장: 저항식 (XPT2046)
- 병실에서 간호사가 장갑을 낀 채 조작할 가능성 높음
- 스타일러스 사용 가능 -> 소독 용이
- SPI 터치이므로 디스플레이 SPI 버스에 CS만 추가하면 됨

---

## 5. 아두이노 라이브러리 지원 현황

| 라이브러리 | ST7796 | RA8875 | SSD1963 | ILI9488 | Sprite | LVGL |
|-----------|--------|--------|---------|---------|--------|------|
| **TFT_eSPI** | 공식 SPI | 미공식 | Parallel만 | 공식 SPI | O | O |
| **LovyanGFX** | 공식 | 개발중 | 미지원 | 공식 | O | O |
| **Adafruit_RA8875** | X | 공식 | X | X | X | X |
| **Adafruit_GFX** | 간접 | 간접 | 간접 | 간접 | X | X |

### TFT_eSPI Sprite 시스템 (파형 표시 핵심)
- ESP32 내부 RAM: 최대 ~200x200 px (16비트) = 약 80KB
- PSRAM 장착 시: 전체 화면 크기 Sprite 가능
- `sprite.scroll()` 메서드로 파형 스크롤 구현
- `pushSprite()` 로 부분 영역만 전송 -> 높은 갱신 속도

---

## 6. 구체적 제품 추천 및 비교표

### 옵션 A: ST7796 3.5인치 SPI (1순위 권장)

| 항목 | 사양 |
|------|------|
| 제품명 | 3.5" IPS SPI LCD ST7796 (480x320) |
| 드라이버 IC | ST7796S |
| 해상도 | 480 x 320 |
| 인터페이스 | SPI (80 MHz 가능) |
| 터치 | 저항식 XPT2046 / 정전식 FT6236 선택 가능 |
| 라이브러리 | TFT_eSPI 공식, LovyanGFX 공식 |
| 구매처 | AliExpress, LCDWiki, Elecrow |
| 가격 | 약 $8~15 (알리), ~15,000원 (국내) |
| 전체화면 FPS | 480x320 @ 80MHz = ~34 FPS |
| 파형 부분갱신 | 480x160 영역 -> ~68 FPS |

**장점:**
- ESP32 생태계에서 가장 안정적이고 검증된 조합
- 80 MHz SPI 클록으로 충분한 refresh rate
- TFT_eSPI Sprite + scroll로 ECG 파형 완벽 구현
- ADS1292R과 SPI 버스 분리 운용 용이
- MISO tristate 정상 동작 (대부분의 보드)
- 가격 대비 최고 효율

**단점:**
- 5인치 요구사항 미충족 (3.5인치)
- 800x480 대비 해상도 낮음

---

### 옵션 B: ST7796 4.0인치 SPI

| 항목 | 사양 |
|------|------|
| 제품명 | MHS-4.0 inch Display-B (RPi Type) ST7796 |
| 드라이버 IC | ST7796S |
| 해상도 | 480 x 320 |
| 인터페이스 | SPI (80 MHz 확인됨) |
| 터치 | 저항식 XPT2046 |
| 라이브러리 | TFT_eSPI 공식 |
| 구매처 | AliExpress "MHS 4.0 inch ST7796" 검색 |
| 가격 | 약 $10~18 (알리) |

**장점:**
- 옵션 A와 동일한 안정성 + 0.5인치 더 큼
- TFT_eSPI에서 "MHS-4.0 inch Display-B" 전용 설정 존재

**단점:**
- 여전히 5인치 미달
- 일부 보드에서 MISO tristate 문제 보고 (다이오드 수정 필요한 경우 있음)

---

### 옵션 C: RA8875 + 5인치 800x480 SPI (5인치 고수 시)

| 항목 | 사양 |
|------|------|
| 제품명 | BuyDisplay ER-TFTM050-3 + RA8875 컨트롤러 보드 |
| 드라이버 IC | RA8875 |
| 해상도 | 800 x 480 |
| 인터페이스 | SPI (최대 ~35 MHz 실측) |
| 터치 | 저항식/정전식 선택 가능 |
| 라이브러리 | Adafruit_RA8875, Roman-Port/RA8875 (ESP32 전용) |
| 구매처 | BuyDisplay.com, icbanq.com |
| 가격 | 약 $30~50 (BuyDisplay), 50,000~70,000원 (국내) |
| 전체화면 FPS | 800x480 @ 35MHz = ~5.7 FPS |
| 파형 부분갱신 | HW 가속 선/사각 그리기로 보상 가능 |

**장점:**
- 5인치 800x480 고해상도
- RA8875 하드웨어 가속 (선 그리기 등) -> SPI 대역폭 한계 보상
- SPI 인터페이스로 핀 절약
- MISO tristate 정상

**단점:**
- TFT_eSPI 공식 미지원 -> Adafruit_RA8875 라이브러리 사용 필요
- Sprite 미지원 -> 파형 flicker-free 구현 난이도 높음
- 전체화면 갱신 매우 느림 (~6 FPS)
- HW 가속에 의존하는 파형 그리기 전략 필요
- 커뮤니티 사례 적음 (ESP32+RA8875 조합)

---

### 옵션 D: SSD1963 + 5인치 800x480 Parallel 8-bit (5인치 + 성능 필요 시)

| 항목 | 사양 |
|------|------|
| 제품명 | 5" 800x480 TFT + SSD1963 Parallel 8-bit |
| 드라이버 IC | SSD1963 |
| 해상도 | 800 x 480 |
| 인터페이스 | Parallel 8-bit (SPI 불가) |
| 터치 | 저항식 XPT2046 |
| 라이브러리 | TFT_eSPI (Setup50_SSD1963_Parallel.h) |
| 구매처 | AliExpress "SSD1963 5 inch 800x480" 검색 |
| 가격 | 약 $15~25 (알리) |
| 벤치마크 | Screen fill ~770ms, 총 ~8초 |

**장점:**
- TFT_eSPI 공식 지원 (Parallel 모드)
- 5인치 800x480 고해상도
- SPI 버스 미사용 -> ADS1292R과 완전 격리

**단점:**
- GPIO 11~12개 소모 (D0-D7 + WR/RD/DC/CS/RST)
- ESP32-WROOM-32D 가용 핀 부족 위험
- Wi-Fi + ADS1292R + SSD1963 + 터치 동시 사용 시 핀 할당 극히 타이트
- Parallel GPIO 토글 속도 한계 (~10 MHz) -> 기대만큼 빠르지 않을 수 있음

---

## 7. 최종 권장 사항

### 1순위 권장: 옵션 A -- ST7796 3.5인치 SPI

**근거:**
1. ESP32 + TFT_eSPI 생태계에서 **가장 검증된 조합** (80 MHz SPI 확인)
2. 480x320 해상도는 ECG 파형 표시에 **충분** (의료 모니터 표준 해상도)
3. ADS1292R과 **SPI 버스 완전 분리** (HSPI/VSPI) 가능
4. Sprite + scroll 기법으로 **flicker-free 실시간 파형** 구현 가능 (부분 갱신 60+ FPS)
5. **가격 최저** (~$10), 즉시 구매 가능
6. 커뮤니티 사례 풍부 -> 문제 해결 용이

### 2순위 대안: 옵션 C -- RA8875 5인치 (5인치 필수 시)

**선택 조건:**
- 화면 크기 5인치가 절대 요구사항인 경우
- RA8875 HW 가속을 활용한 파형 그리기에 추가 개발 시간 투자 가능한 경우
- TFT_eSPI Sprite 미지원을 감수할 수 있는 경우

### 권장하지 않는 선택

- **ILI9488**: MISO tristate 미지원 -> ADS1292R SPI 버스 충돌 **치명적**
- **ILI9806**: Arduino/ESP32 라이브러리 사실상 미존재
- **SSD1963 Parallel**: GPIO 소모 과다 -> 핀 부족 위험

---

## 8. 프로젝트 적용 계획

### Phase 1 (기획서 기준)에서의 적용

1. **부품 구매**: ST7796 3.5인치 SPI LCD (저항식 터치 포함) -- AliExpress 또는 icbanq
2. **핀 할당**:
   - HSPI (SPI2) -> ADS1292R 전용
   - VSPI (SPI3) -> 디스플레이 + 터치 전용
3. **소프트웨어**:
   - TFT_eSPI 라이브러리 + ST7796_DRIVER 설정
   - Sprite 기반 ECG 파형 스크롤 구현
   - LVGL 연동으로 UI 구성 (심박수, 알림 등)
4. **동시 구동 검증**: EKG 측정 + 디스플레이 + Wi-Fi 동시 동작

### 주의사항
- ch07-hardware.md의 핀 할당(GPIO 18/23/19/5)이 VSPI 기본값 -> 디스플레이와 충돌
- **ADS1292R을 HSPI로 이동** 필요 (GPIO 14/13/12/15)
- 기존 ch07 핀맵 업데이트 필요 (hardware-circuit-reviewer에게 위임)

---

## Sources

- [ESP32 Forum - TFT SPI at 60MHz](https://www.esp32.com/viewtopic.php?t=6343)
- [ESP32 Forum - 80MHz SPI speed](https://www.esp32.com/viewtopic.php?t=6627)
- [TFT_eSPI GitHub Repository](https://github.com/Bodmer/TFT_eSPI)
- [TFT_eSPI - ST7796 and ILI9488 MISO Warning](https://github.com/Bodmer/TFT_eSPI/discussions/898)
- [TFT_eSPI - SSD1963 vs SPI Speed](https://github.com/Bodmer/TFT_eSPI/discussions/1075)
- [TFT_eSPI - SSD1963 5inch Setup](https://github.com/Bodmer/TFT_eSPI/discussions/3768)
- [LovyanGFX GitHub](https://github.com/lovyan03/LovyanGFX)
- [LovyanGFX RA8875 Issue](https://github.com/lovyan03/LovyanGFX/issues/52)
- [BuyDisplay 5" RA8875 800x480](https://www.buydisplay.com/5-inch-tft-lcd-module-800x480-display-controller-i2c-serial-spi)
- [BuyDisplay 5" Capacitive Touch RA8875](https://www.buydisplay.com/5-inch-tft-lcd-display-capacitive-touchscreen-ra8875-controller-800x480)
- [Adafruit RA8875 Driver Board](https://www.adafruit.com/product/1590)
- [RA8875 Datasheet (PDF)](https://cdn-shop.adafruit.com/datasheets/RA8875_DS_V19_Eng.pdf)
- [Roman-Port/RA8875 ESP32 Driver](https://github.com/Roman-Port/RA8875)
- [ESP32 SPI Communication Tutorial](https://randomnerdtutorials.com/esp32-spi-communication-arduino/)
- [Espressif LCD FAQ](https://docs.espressif.com/projects/esp-faq/en/latest/software-framework/peripherals/lcd.html)
- [ESP32 Touch Panel Documentation](https://espressif-docs.readthedocs-hosted.com/projects/espressif-esp-iot-solution/en/latest/input_device/touch_panel.html)
- [TFT_eSPI Sprite System (DeepWiki)](https://deepwiki.com/Bodmer/TFT_eSPI/6.1-sprite-system)
- [Elecrow 3.5" IPS SPI ST7796](https://www.elecrow.com/3-5-ips-spi-lcd-capacitive-touch-module-st7796-driver-320-480-resolution.html)
- [LCDWiki 3.5" IPS SPI ST7796](https://www.lcdwiki.com/3.5inch_IPS_SPI_Module_ST7796)
- [Arduino Forum - ECG TFT UI Source](https://forum.arduino.cc/t/ooking-for-ecg-tft-ui-source-esp32-arduino-ad8232/1420399)
