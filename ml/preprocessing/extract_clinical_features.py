"""
환자별 임상 진단/처방 플래그 추출

MIMIC-IV DB에서 4개 이진 플래그를 환자별로 추출하여 HDF5에 저장:
  - DM (당뇨): ICD-10 E10/E11, ICD-9 250
  - HF (심부전): ICD-10 I50, ICD-9 428
  - MI (심근경색): ICD-10 I21, ICD-9 410
  - AHT (항고혈압제): prescriptions 테이블에서 약물명 매칭

출력: clinical_features.h5
  - subject_id: (N_patients,) int64
  - clinical_flags: (N_patients, 4) int8 — [DM, HF, MI, AHT]
"""
import time
import numpy as np
import h5py
import psycopg2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import DB_CONFIG

INPUT_H5 = "g:/AIEKG/ml/data/ecg_preprocessed.h5"
OUTPUT_H5 = "g:/AIEKG/ml/data/clinical_features.h5"

# 항고혈압제 generic names (주요 클래스)
AHT_MEDICATIONS = [
    # ACE inhibitors
    "lisinopril", "enalapril", "ramipril", "captopril", "benazepril",
    "fosinopril", "quinapril", "perindopril", "trandolapril",
    # ARBs
    "losartan", "valsartan", "olmesartan", "irbesartan", "candesartan",
    "telmisartan", "azilsartan",
    # Calcium channel blockers
    "amlodipine", "nifedipine", "diltiazem", "verapamil", "felodipine",
    # Beta blockers
    "metoprolol", "atenolol", "carvedilol", "bisoprolol", "propranolol",
    "labetalol", "nebivolol",
    # Thiazide diuretics
    "hydrochlorothiazide", "chlorthalidone", "indapamide",
    # Other diuretics (loop)
    "furosemide", "bumetanide", "torsemide",
    # Aldosterone antagonists
    "spironolactone", "eplerenone",
]


def get_unique_patients(h5_path):
    """HDF5에서 고유 환자 ID 추출"""
    with h5py.File(h5_path, "r") as f:
        subject_ids = f["subject_id"][:]
    return np.unique(subject_ids)


def fetch_diagnosis_flags(conn, patient_ids, icd10_prefix, icd9_prefix):
    """특정 ICD 코드를 가진 환자 집합 반환"""
    cur = conn.cursor()

    # LIKE 패턴 생성 (ICD-10은 여러 prefix 가능)
    conditions = []
    params = []

    if isinstance(icd10_prefix, list):
        for prefix in icd10_prefix:
            conditions.append("(d.icd_code LIKE %s AND d.icd_version = 10)")
            params.append(f"{prefix}%")
    else:
        conditions.append("(d.icd_code LIKE %s AND d.icd_version = 10)")
        params.append(f"{icd10_prefix}%")

    conditions.append("(d.icd_code LIKE %s AND d.icd_version = 9)")
    params.append(f"{icd9_prefix}%")

    where_clause = " OR ".join(conditions)

    cur.execute(f"""
        SELECT DISTINCT d.subject_id
        FROM mimiciv_hosp.diagnoses_icd d
        WHERE ({where_clause})
        AND d.subject_id = ANY(%s)
    """, (*params, list(patient_ids)))

    rows = cur.fetchall()
    cur.close()
    return set(r[0] for r in rows)


def fetch_aht_patients(conn, patient_ids):
    """항고혈압제 처방 이력이 있는 환자 집합 반환"""
    # SIMILAR TO 패턴으로 약물명 매칭
    pattern = "%(" + "|".join(AHT_MEDICATIONS) + ")%"

    batch_size = 10000
    aht_patients = set()

    for i in range(0, len(patient_ids), batch_size):
        batch = patient_ids[i:i + batch_size]
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT subject_id
            FROM mimiciv_hosp.prescriptions
            WHERE LOWER(drug) SIMILAR TO %s
            AND subject_id = ANY(%s)
        """, (pattern, list(batch)))
        rows = cur.fetchall()
        cur.close()
        aht_patients.update(r[0] for r in rows)
        print(f"    AHT 배치 {i // batch_size + 1}/"
              f"{(len(patient_ids) + batch_size - 1) // batch_size}: "
              f"누적 {len(aht_patients):,}명")

    return aht_patients


def main():
    print("[1/3] 환자 목록 로드...")
    unique_patients = get_unique_patients(INPUT_H5)
    patient_list = [int(x) for x in unique_patients]
    print(f"  고유 환자: {len(patient_list):,}명")

    print("\n[2/3] DB에서 임상 플래그 추출...")
    conn = psycopg2.connect(**DB_CONFIG)
    t0 = time.time()

    # DM (당뇨)
    print("  DM (당뇨) 추출 중...")
    dm_patients = fetch_diagnosis_flags(conn, patient_list, ["E10", "E11"], "250")
    print(f"    DM 환자: {len(dm_patients):,}명 ({len(dm_patients)/len(patient_list)*100:.1f}%)")

    # HF (심부전)
    print("  HF (심부전) 추출 중...")
    hf_patients = fetch_diagnosis_flags(conn, patient_list, "I50", "428")
    print(f"    HF 환자: {len(hf_patients):,}명 ({len(hf_patients)/len(patient_list)*100:.1f}%)")

    # MI (심근경색)
    print("  MI (심근경색) 추출 중...")
    mi_patients = fetch_diagnosis_flags(conn, patient_list, "I21", "410")
    print(f"    MI 환자: {len(mi_patients):,}명 ({len(mi_patients)/len(patient_list)*100:.1f}%)")

    # AHT (항고혈압제)
    print("  AHT (항고혈압제) 추출 중...")
    aht_patients = fetch_aht_patients(conn, patient_list)
    print(f"    AHT 환자: {len(aht_patients):,}명 ({len(aht_patients)/len(patient_list)*100:.1f}%)")

    conn.close()
    print(f"  DB 추출 완료 ({time.time() - t0:.0f}s)")

    # 플래그 배열 생성
    clinical_flags = np.zeros((len(unique_patients), 4), dtype=np.int8)
    for i, sid in enumerate(unique_patients):
        sid_int = int(sid)
        if sid_int in dm_patients:
            clinical_flags[i, 0] = 1
        if sid_int in hf_patients:
            clinical_flags[i, 1] = 1
        if sid_int in mi_patients:
            clinical_flags[i, 2] = 1
        if sid_int in aht_patients:
            clinical_flags[i, 3] = 1

    # 통계
    any_flag = np.any(clinical_flags > 0, axis=1).sum()
    print(f"\n  최소 1개 플래그 보유: {any_flag:,}명 ({any_flag/len(unique_patients)*100:.1f}%)")
    print(f"  DM+HF 동시: {np.sum((clinical_flags[:, 0] == 1) & (clinical_flags[:, 1] == 1)):,}명")

    # 저장
    print(f"\n[3/3] HDF5 저장: {OUTPUT_H5}")
    with h5py.File(OUTPUT_H5, "w") as f:
        f.create_dataset("subject_id", data=unique_patients)
        f.create_dataset("clinical_flags", data=clinical_flags)
        f.attrs["feature_names"] = ["dm", "hf", "mi", "aht"]

    print(f"  환자: {len(unique_patients):,}, 피처: {clinical_flags.shape[1]}")
    print("  완료!")


if __name__ == "__main__":
    main()
