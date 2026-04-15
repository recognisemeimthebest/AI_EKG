"""
DB에서 환자 피처 (age, gender) 추출
"""
import pandas as pd


def load_patient_features(db_config: dict) -> pd.DataFrame:
    """
    mimiciv_hosp.patients 테이블에서 age, gender 추출.
    Returns: DataFrame [subject_id, age, gender_code]
    """
    import psycopg2

    conn = psycopg2.connect(**db_config)
    query = """
    SELECT subject_id, anchor_age AS age, gender
    FROM mimiciv_hosp.patients
    """
    df = pd.read_sql(query, conn)
    conn.close()

    # gender 인코딩: M=0, F=1
    df["gender_code"] = (df["gender"] == "F").astype(int)
    df = df[["subject_id", "age", "gender_code"]]
    return df


def load_patient_features_csv(patients_csv: str) -> pd.DataFrame:
    """
    DB 없이 CSV에서 직접 로드 (fallback).
    """
    df = pd.read_csv(patients_csv, usecols=["subject_id", "anchor_age", "gender"])
    df = df.rename(columns={"anchor_age": "age"})
    df["gender_code"] = (df["gender"] == "F").astype(int)
    df = df[["subject_id", "age", "gender_code"]]
    return df
