"""
MIMIC-IV ECG report_0 라벨 분포 분석 및 3-class 매핑 정의
"""
import pandas as pd
from collections import Counter

# machine_measurements.csv 로드
CSV_PATH = "g:/AIEKG/data/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/machine_measurements.csv"

print("=" * 60)
print("MIMIC-IV ECG report_0 라벨 분포 분석")
print("=" * 60)

df = pd.read_csv(CSV_PATH, usecols=["subject_id", "study_id", "report_0"])
print(f"\n총 ECG 레코드 수: {len(df):,}")
print(f"결측치 (report_0 없음): {df['report_0'].isna().sum():,}")

# report_0 분포
counts = df["report_0"].value_counts()
print(f"\n고유 라벨 수: {len(counts)}")
print("\n--- report_0 전체 분포 ---")
for label, count in counts.items():
    pct = count / len(df) * 100
    print(f"  {label:<45} {count:>8,}  ({pct:5.1f}%)")

# 3-class 매핑 정의
LABEL_MAP = {
    # Normal (Class 0)
    "Sinus rhythm": 0,
    "Normal sinus rhythm": 0,
    # AFib (Class 1)
    "Atrial fibrillation": 1,
    "Atrial flutter": 1,
    # Other (Class 2) - 나머지 전부
}
CLASS_NAMES = {0: "Normal", 1: "AFib", 2: "Other"}

def map_label(report_0):
    if pd.isna(report_0):
        return -1  # 결측
    return LABEL_MAP.get(report_0, 2)  # 매핑에 없으면 Other

df["label"] = df["report_0"].apply(map_label)

print("\n--- 3-class 매핑 결과 ---")
for cls_id, cls_name in CLASS_NAMES.items():
    count = (df["label"] == cls_id).sum()
    pct = count / len(df) * 100
    print(f"  Class {cls_id} ({cls_name:<8}): {count:>8,}  ({pct:5.1f}%)")

excluded = (df["label"] == -1).sum()
print(f"  제외 (결측)         : {excluded:>8,}  ({excluded/len(df)*100:5.1f}%)")

# Other 클래스 내부 분포도 확인
print("\n--- Other 클래스 내부 분포 (상위 10) ---")
other_df = df[df["label"] == 2]
other_counts = other_df["report_0"].value_counts().head(10)
for label, count in other_counts.items():
    print(f"  {label:<45} {count:>8,}")
