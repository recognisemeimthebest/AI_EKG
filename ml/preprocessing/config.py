"""
전처리 파이프라인 설정값
"""
from pathlib import Path

# ======================== 경로 ========================
PROJECT_ROOT = Path("g:/AIEKG")
ECG_DATA_DIR = PROJECT_ROOT / "data/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0"
ECG_FILES_DIR = ECG_DATA_DIR / "files"
RECORD_LIST_CSV = ECG_DATA_DIR / "record_list.csv"
MACHINE_MEAS_CSV = ECG_DATA_DIR / "machine_measurements.csv"
OUTPUT_DIR = PROJECT_ROOT / "ml/data"

# ======================== DB ========================
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "mimic4",
    "user": "postgres",
    "password": "tlsghktk6",
}

# ======================== ECG 파라미터 ========================
SAMPLE_RATE = 500       # Hz
DURATION = 10           # seconds
N_SAMPLES = SAMPLE_RATE * DURATION  # 5000
N_LEADS = 12
LEAD_NAMES = ["I", "II", "III", "aVR", "aVF", "aVL", "V1", "V2", "V3", "V4", "V5", "V6"]

# ======================== 필터링 ========================
HIGHPASS_FREQ = 0.5     # Hz - 베이스라인 원더 제거
LOWPASS_FREQ = 50.0     # Hz - 고주파 노이즈 제거
NOTCH_FREQ = 60.0       # Hz - 전원 노이즈 제거
FILTER_ORDER = 4        # Butterworth 필터 차수

# ======================== 3-class 라벨 매핑 ========================
# 키워드 기반 매핑 (소문자 변환 후 매칭, 마침표/대소문자 무관)
# 우선순위: AFib > Normal > Other (AFib 키워드가 있으면 AFib)
AFIB_KEYWORDS = [
    "atrial fibrillation",
    "atrial flutter",
]
NORMAL_KEYWORDS = [
    "sinus rhythm",
    "normal sinus rhythm",
    "sinus arrhythmia",       # 정상 변이
]
# 위 두 카테고리에 해당하지 않으면 Other (class 2)

CLASS_NAMES = {0: "Normal", 1: "AFib", 2: "Other"}
N_CLASSES = 3

# ======================== 수치 피처 ========================
NUMERIC_FEATURES = ["rr_interval", "qrs_duration", "p_duration", "qrs_axis"]
PATIENT_FEATURES = ["age", "gender"]  # gender: M=0, F=1

# ======================== 품질 필터링 ========================
# 이 키워드가 report_0에 있으면 제외
EXCLUDE_KEYWORDS = [
    "data quality",
    "pacer detection suspended",
    "pediatric criteria",
    "age not entered",
]
