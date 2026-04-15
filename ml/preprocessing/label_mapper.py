"""
report_0 텍스트 → 3-class 라벨 매핑
키워드 기반으로 마침표/대소문자 변형을 모두 처리
"""
import pandas as pd
from config import AFIB_KEYWORDS, NORMAL_KEYWORDS, EXCLUDE_KEYWORDS, CLASS_NAMES


def map_report_to_label(report_0: str) -> int:
    """
    report_0 텍스트를 3-class 라벨로 변환.
    Returns: 0=Normal, 1=AFib, 2=Other, -1=제외
    """
    if pd.isna(report_0):
        return -1

    text = report_0.strip().rstrip(".").lower()

    # 품질 이슈 데이터 제외
    for kw in EXCLUDE_KEYWORDS:
        if kw in text:
            return -1

    # AFib 우선 (Sinus rhythm with AFib 같은 케이스 방지)
    for kw in AFIB_KEYWORDS:
        if kw in text:
            return 1

    # Normal
    for kw in NORMAL_KEYWORDS:
        if kw in text:
            return 0

    # 나머지는 Other
    return 2


def build_label_dataframe(machine_meas_csv: str) -> pd.DataFrame:
    """
    machine_measurements.csv에서 라벨 + 수치 피처 추출.
    Returns: DataFrame with columns [subject_id, study_id, label, rr_interval, qrs_duration, p_duration, qrs_axis]
    """
    cols = ["subject_id", "study_id", "report_0",
            "rr_interval", "p_onset", "p_end", "qrs_onset", "qrs_end", "qrs_axis"]
    df = pd.read_csv(machine_meas_csv, usecols=cols)

    # 라벨 매핑
    df["label"] = df["report_0"].apply(map_report_to_label)

    # 파생 피처 계산
    df["qrs_duration"] = df["qrs_end"] - df["qrs_onset"]
    df["p_duration"] = df["p_end"] - df["p_onset"]

    # 제외 데이터 필터링
    df = df[df["label"] != -1].copy()

    # 필요한 컬럼만
    result = df[["subject_id", "study_id", "label",
                 "rr_interval", "qrs_duration", "p_duration", "qrs_axis"]].copy()

    return result


if __name__ == "__main__":
    from config import MACHINE_MEAS_CSV

    df = build_label_dataframe(str(MACHINE_MEAS_CSV))
    print(f"유효 레코드: {len(df):,}")
    print(f"\n클래스 분포:")
    for cls_id, cls_name in CLASS_NAMES.items():
        count = (df["label"] == cls_id).sum()
        pct = count / len(df) * 100
        print(f"  {cls_id} ({cls_name:<8}): {count:>8,} ({pct:5.1f}%)")

    print(f"\n수치 피처 통계:")
    for col in ["rr_interval", "qrs_duration", "p_duration", "qrs_axis"]:
        print(f"  {col}: mean={df[col].mean():.1f}, std={df[col].std():.1f}, "
              f"null={df[col].isna().sum():,}")
