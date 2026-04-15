-- =============================================
-- MIMIC-IV 테이블/컬럼 한글 코멘트
-- =============================================

-- =====================
-- mimiciv_hosp 스키마
-- =====================

COMMENT ON SCHEMA mimiciv_hosp IS '병원 정보 (입퇴원, 진단, 검사, 투약 등)';

-- patients
COMMENT ON TABLE mimiciv_hosp.patients IS '환자 기본 정보';
COMMENT ON COLUMN mimiciv_hosp.patients.subject_id IS '환자 고유 ID';
COMMENT ON COLUMN mimiciv_hosp.patients.gender IS '성별 (M: 남성, F: 여성)';
COMMENT ON COLUMN mimiciv_hosp.patients.anchor_age IS '기준 나이';
COMMENT ON COLUMN mimiciv_hosp.patients.anchor_year IS '기준 연도';
COMMENT ON COLUMN mimiciv_hosp.patients.anchor_year_group IS '기준 연도 그룹';
COMMENT ON COLUMN mimiciv_hosp.patients.dod IS '사망일';

-- admissions
COMMENT ON TABLE mimiciv_hosp.admissions IS '입퇴원 기록';
COMMENT ON COLUMN mimiciv_hosp.admissions.subject_id IS '환자 고유 ID';
COMMENT ON COLUMN mimiciv_hosp.admissions.hadm_id IS '입원 고유 ID';
COMMENT ON COLUMN mimiciv_hosp.admissions.admittime IS '입원 일시';
COMMENT ON COLUMN mimiciv_hosp.admissions.dischtime IS '퇴원 일시';
COMMENT ON COLUMN mimiciv_hosp.admissions.deathtime IS '사망 일시';
COMMENT ON COLUMN mimiciv_hosp.admissions.admission_type IS '입원 유형 (응급, 선택적 등)';
COMMENT ON COLUMN mimiciv_hosp.admissions.admit_provider_id IS '담당 의료진 ID';
COMMENT ON COLUMN mimiciv_hosp.admissions.admission_location IS '입원 경로 (응급실, 외래 등)';
COMMENT ON COLUMN mimiciv_hosp.admissions.discharge_location IS '퇴원 행선지 (자택, 재활시설 등)';
COMMENT ON COLUMN mimiciv_hosp.admissions.insurance IS '보험 종류';
COMMENT ON COLUMN mimiciv_hosp.admissions.language IS '사용 언어';
COMMENT ON COLUMN mimiciv_hosp.admissions.marital_status IS '결혼 상태';
COMMENT ON COLUMN mimiciv_hosp.admissions.race IS '인종';
COMMENT ON COLUMN mimiciv_hosp.admissions.edregtime IS '응급실 등록 일시';
COMMENT ON COLUMN mimiciv_hosp.admissions.edouttime IS '응급실 퇴실 일시';
COMMENT ON COLUMN mimiciv_hosp.admissions.hospital_expire_flag IS '원내 사망 여부 (1: 사망)';

-- d_hcpcs
COMMENT ON TABLE mimiciv_hosp.d_hcpcs IS 'HCPCS 코드 사전 (의료 행위/장비 코드)';
COMMENT ON COLUMN mimiciv_hosp.d_hcpcs.code IS 'HCPCS 코드';
COMMENT ON COLUMN mimiciv_hosp.d_hcpcs.category IS '분류 번호';
COMMENT ON COLUMN mimiciv_hosp.d_hcpcs.long_description IS '상세 설명';
COMMENT ON COLUMN mimiciv_hosp.d_hcpcs.short_description IS '간략 설명';

-- d_icd_diagnoses
COMMENT ON TABLE mimiciv_hosp.d_icd_diagnoses IS 'ICD 진단 코드 사전';
COMMENT ON COLUMN mimiciv_hosp.d_icd_diagnoses.icd_code IS 'ICD 진단 코드';
COMMENT ON COLUMN mimiciv_hosp.d_icd_diagnoses.icd_version IS 'ICD 버전 (9 또는 10)';
COMMENT ON COLUMN mimiciv_hosp.d_icd_diagnoses.long_title IS '진단명';

-- d_icd_procedures
COMMENT ON TABLE mimiciv_hosp.d_icd_procedures IS 'ICD 시술/수술 코드 사전';
COMMENT ON COLUMN mimiciv_hosp.d_icd_procedures.icd_code IS 'ICD 시술 코드';
COMMENT ON COLUMN mimiciv_hosp.d_icd_procedures.icd_version IS 'ICD 버전 (9 또는 10)';
COMMENT ON COLUMN mimiciv_hosp.d_icd_procedures.long_title IS '시술/수술명';

-- d_labitems
COMMENT ON TABLE mimiciv_hosp.d_labitems IS '검사 항목 사전';
COMMENT ON COLUMN mimiciv_hosp.d_labitems.itemid IS '검사 항목 ID';
COMMENT ON COLUMN mimiciv_hosp.d_labitems.label IS '검사 항목명';
COMMENT ON COLUMN mimiciv_hosp.d_labitems.fluid IS '검체 종류 (혈액, 소변 등)';
COMMENT ON COLUMN mimiciv_hosp.d_labitems.category IS '검사 분류';

-- diagnoses_icd
COMMENT ON TABLE mimiciv_hosp.diagnoses_icd IS '환자별 ICD 진단 기록';
COMMENT ON COLUMN mimiciv_hosp.diagnoses_icd.subject_id IS '환자 고유 ID';
COMMENT ON COLUMN mimiciv_hosp.diagnoses_icd.hadm_id IS '입원 고유 ID';
COMMENT ON COLUMN mimiciv_hosp.diagnoses_icd.seq_num IS '진단 우선순위 (1이 주진단)';
COMMENT ON COLUMN mimiciv_hosp.diagnoses_icd.icd_code IS 'ICD 진단 코드';
COMMENT ON COLUMN mimiciv_hosp.diagnoses_icd.icd_version IS 'ICD 버전 (9 또는 10)';

-- drgcodes
COMMENT ON TABLE mimiciv_hosp.drgcodes IS 'DRG 코드 (진단관련군, 보험 청구용)';
COMMENT ON COLUMN mimiciv_hosp.drgcodes.subject_id IS '환자 고유 ID';
COMMENT ON COLUMN mimiciv_hosp.drgcodes.hadm_id IS '입원 고유 ID';
COMMENT ON COLUMN mimiciv_hosp.drgcodes.drg_type IS 'DRG 유형 (HCFA, APR)';
COMMENT ON COLUMN mimiciv_hosp.drgcodes.drg_code IS 'DRG 코드';
COMMENT ON COLUMN mimiciv_hosp.drgcodes.description IS 'DRG 설명';
COMMENT ON COLUMN mimiciv_hosp.drgcodes.drg_severity IS '질병 중증도';
COMMENT ON COLUMN mimiciv_hosp.drgcodes.drg_mortality IS '사망 위험도';

-- emar
COMMENT ON TABLE mimiciv_hosp.emar IS '전자 투약 기록 (eMAR)';
COMMENT ON COLUMN mimiciv_hosp.emar.subject_id IS '환자 고유 ID';
COMMENT ON COLUMN mimiciv_hosp.emar.hadm_id IS '입원 고유 ID';
COMMENT ON COLUMN mimiciv_hosp.emar.emar_id IS '투약 기록 ID';
COMMENT ON COLUMN mimiciv_hosp.emar.emar_seq IS '투약 순번';
COMMENT ON COLUMN mimiciv_hosp.emar.poe_id IS '처방 ID';
COMMENT ON COLUMN mimiciv_hosp.emar.pharmacy_id IS '약국 처방 ID';
COMMENT ON COLUMN mimiciv_hosp.emar.enter_provider_id IS '입력 의료진 ID';
COMMENT ON COLUMN mimiciv_hosp.emar.charttime IS '투약 기록 일시';
COMMENT ON COLUMN mimiciv_hosp.emar.medication IS '약물명';
COMMENT ON COLUMN mimiciv_hosp.emar.event_txt IS '투약 이벤트 (투여됨, 보류 등)';
COMMENT ON COLUMN mimiciv_hosp.emar.scheduletime IS '예정 투약 일시';
COMMENT ON COLUMN mimiciv_hosp.emar.storetime IS '시스템 저장 일시';

-- emar_detail
COMMENT ON TABLE mimiciv_hosp.emar_detail IS '투약 상세 기록 (용량, 경로, 주입속도 등)';
COMMENT ON COLUMN mimiciv_hosp.emar_detail.subject_id IS '환자 고유 ID';
COMMENT ON COLUMN mimiciv_hosp.emar_detail.emar_id IS '투약 기록 ID';
COMMENT ON COLUMN mimiciv_hosp.emar_detail.emar_seq IS '투약 순번';
COMMENT ON COLUMN mimiciv_hosp.emar_detail.parent_field_ordinal IS '상위 필드 순서';
COMMENT ON COLUMN mimiciv_hosp.emar_detail.administration_type IS '투여 방식';
COMMENT ON COLUMN mimiciv_hosp.emar_detail.pharmacy_id IS '약국 처방 ID';
COMMENT ON COLUMN mimiciv_hosp.emar_detail.barcode_type IS '바코드 유형';
COMMENT ON COLUMN mimiciv_hosp.emar_detail.reason_for_no_barcode IS '바코드 미사용 사유';
COMMENT ON COLUMN mimiciv_hosp.emar_detail.complete_dose_not_given IS '전량 미투여 여부';
COMMENT ON COLUMN mimiciv_hosp.emar_detail.dose_due IS '예정 용량';
COMMENT ON COLUMN mimiciv_hosp.emar_detail.dose_due_unit IS '예정 용량 단위';
COMMENT ON COLUMN mimiciv_hosp.emar_detail.dose_given IS '실제 투여 용량';
COMMENT ON COLUMN mimiciv_hosp.emar_detail.dose_given_unit IS '실제 투여 단위';
COMMENT ON COLUMN mimiciv_hosp.emar_detail.will_remainder_of_dose_be_given IS '잔여 용량 투여 예정 여부';
COMMENT ON COLUMN mimiciv_hosp.emar_detail.product_amount_given IS '제품 투여량';
COMMENT ON COLUMN mimiciv_hosp.emar_detail.product_unit IS '제품 단위';
COMMENT ON COLUMN mimiciv_hosp.emar_detail.product_code IS '제품 코드';
COMMENT ON COLUMN mimiciv_hosp.emar_detail.product_description IS '제품 설명';
COMMENT ON COLUMN mimiciv_hosp.emar_detail.product_description_other IS '기타 제품 설명';
COMMENT ON COLUMN mimiciv_hosp.emar_detail.prior_infusion_rate IS '이전 주입 속도';
COMMENT ON COLUMN mimiciv_hosp.emar_detail.infusion_rate IS '주입 속도';
COMMENT ON COLUMN mimiciv_hosp.emar_detail.infusion_rate_adjustment IS '주입 속도 조정 내용';
COMMENT ON COLUMN mimiciv_hosp.emar_detail.infusion_rate_adjustment_amount IS '주입 속도 조정량';
COMMENT ON COLUMN mimiciv_hosp.emar_detail.infusion_rate_unit IS '주입 속도 단위';
COMMENT ON COLUMN mimiciv_hosp.emar_detail.route IS '투여 경로 (경구, 정맥 등)';
COMMENT ON COLUMN mimiciv_hosp.emar_detail.infusion_complete IS '주입 완료 여부';
COMMENT ON COLUMN mimiciv_hosp.emar_detail.completion_interval IS '완료 간격';
COMMENT ON COLUMN mimiciv_hosp.emar_detail.new_iv_bag_hung IS '새 IV 백 교체 여부';
COMMENT ON COLUMN mimiciv_hosp.emar_detail.continued_infusion_in_other_location IS '타 장소 주입 계속 여부';
COMMENT ON COLUMN mimiciv_hosp.emar_detail.restart_interval IS '재시작 간격';
COMMENT ON COLUMN mimiciv_hosp.emar_detail.side IS '투여 부위 (좌/우)';
COMMENT ON COLUMN mimiciv_hosp.emar_detail.site IS '투여 위치 상세';
COMMENT ON COLUMN mimiciv_hosp.emar_detail.non_formulary_visual_verification IS '비처방집 약물 육안 확인 여부';

-- hcpcsevents
COMMENT ON TABLE mimiciv_hosp.hcpcsevents IS 'HCPCS 의료 행위 기록';
COMMENT ON COLUMN mimiciv_hosp.hcpcsevents.subject_id IS '환자 고유 ID';
COMMENT ON COLUMN mimiciv_hosp.hcpcsevents.hadm_id IS '입원 고유 ID';
COMMENT ON COLUMN mimiciv_hosp.hcpcsevents.chartdate IS '기록 날짜';
COMMENT ON COLUMN mimiciv_hosp.hcpcsevents.hcpcs_cd IS 'HCPCS 코드';
COMMENT ON COLUMN mimiciv_hosp.hcpcsevents.seq_num IS '순번';
COMMENT ON COLUMN mimiciv_hosp.hcpcsevents.short_description IS '간략 설명';

-- labevents
COMMENT ON TABLE mimiciv_hosp.labevents IS '검사 결과 (혈액검사, 소변검사 등)';
COMMENT ON COLUMN mimiciv_hosp.labevents.labevent_id IS '검사 결과 고유 ID';
COMMENT ON COLUMN mimiciv_hosp.labevents.subject_id IS '환자 고유 ID';
COMMENT ON COLUMN mimiciv_hosp.labevents.hadm_id IS '입원 고유 ID';
COMMENT ON COLUMN mimiciv_hosp.labevents.specimen_id IS '검체 ID';
COMMENT ON COLUMN mimiciv_hosp.labevents.itemid IS '검사 항목 ID (d_labitems 참조)';
COMMENT ON COLUMN mimiciv_hosp.labevents.order_provider_id IS '처방 의료진 ID';
COMMENT ON COLUMN mimiciv_hosp.labevents.charttime IS '검체 채취 일시';
COMMENT ON COLUMN mimiciv_hosp.labevents.storetime IS '시스템 저장 일시';
COMMENT ON COLUMN mimiciv_hosp.labevents.value IS '검사 결과값 (문자열)';
COMMENT ON COLUMN mimiciv_hosp.labevents.valuenum IS '검사 결과값 (숫자)';
COMMENT ON COLUMN mimiciv_hosp.labevents.valueuom IS '결과 단위';
COMMENT ON COLUMN mimiciv_hosp.labevents.ref_range_lower IS '정상 범위 하한';
COMMENT ON COLUMN mimiciv_hosp.labevents.ref_range_upper IS '정상 범위 상한';
COMMENT ON COLUMN mimiciv_hosp.labevents.flag IS '이상 여부 (abnormal 등)';
COMMENT ON COLUMN mimiciv_hosp.labevents.priority IS '검사 우선순위 (ROUTINE, STAT)';
COMMENT ON COLUMN mimiciv_hosp.labevents.comments IS '비고';

-- microbiologyevents
COMMENT ON TABLE mimiciv_hosp.microbiologyevents IS '미생물 배양 검사 결과';
COMMENT ON COLUMN mimiciv_hosp.microbiologyevents.microevent_id IS '미생물 검사 고유 ID';
COMMENT ON COLUMN mimiciv_hosp.microbiologyevents.subject_id IS '환자 고유 ID';
COMMENT ON COLUMN mimiciv_hosp.microbiologyevents.hadm_id IS '입원 고유 ID';
COMMENT ON COLUMN mimiciv_hosp.microbiologyevents.micro_specimen_id IS '미생물 검체 ID';
COMMENT ON COLUMN mimiciv_hosp.microbiologyevents.order_provider_id IS '처방 의료진 ID';
COMMENT ON COLUMN mimiciv_hosp.microbiologyevents.chartdate IS '기록 날짜';
COMMENT ON COLUMN mimiciv_hosp.microbiologyevents.charttime IS '기록 일시';
COMMENT ON COLUMN mimiciv_hosp.microbiologyevents.spec_itemid IS '검체 종류 ID';
COMMENT ON COLUMN mimiciv_hosp.microbiologyevents.spec_type_desc IS '검체 종류명 (혈액, 소변 등)';
COMMENT ON COLUMN mimiciv_hosp.microbiologyevents.test_seq IS '검사 순번';
COMMENT ON COLUMN mimiciv_hosp.microbiologyevents.storedate IS '저장 날짜';
COMMENT ON COLUMN mimiciv_hosp.microbiologyevents.storetime IS '저장 일시';
COMMENT ON COLUMN mimiciv_hosp.microbiologyevents.test_itemid IS '검사 항목 ID';
COMMENT ON COLUMN mimiciv_hosp.microbiologyevents.test_name IS '검사명';
COMMENT ON COLUMN mimiciv_hosp.microbiologyevents.org_itemid IS '검출 균종 ID';
COMMENT ON COLUMN mimiciv_hosp.microbiologyevents.org_name IS '검출 균종명';
COMMENT ON COLUMN mimiciv_hosp.microbiologyevents.isolate_num IS '분리균 번호';
COMMENT ON COLUMN mimiciv_hosp.microbiologyevents.quantity IS '균 수량';
COMMENT ON COLUMN mimiciv_hosp.microbiologyevents.ab_itemid IS '항생제 ID';
COMMENT ON COLUMN mimiciv_hosp.microbiologyevents.ab_name IS '항생제명';
COMMENT ON COLUMN mimiciv_hosp.microbiologyevents.dilution_text IS '희석 배수 텍스트';
COMMENT ON COLUMN mimiciv_hosp.microbiologyevents.dilution_comparison IS '희석 비교 연산자';
COMMENT ON COLUMN mimiciv_hosp.microbiologyevents.dilution_value IS '희석 배수 값';
COMMENT ON COLUMN mimiciv_hosp.microbiologyevents.interpretation IS '감수성 결과 (S: 감수성, R: 내성, I: 중간)';
COMMENT ON COLUMN mimiciv_hosp.microbiologyevents.comments IS '비고';

-- omr
COMMENT ON TABLE mimiciv_hosp.omr IS '외래 의무 기록 (체중, 혈압, BMI 등)';
COMMENT ON COLUMN mimiciv_hosp.omr.subject_id IS '환자 고유 ID';
COMMENT ON COLUMN mimiciv_hosp.omr.chartdate IS '기록 날짜';
COMMENT ON COLUMN mimiciv_hosp.omr.seq_num IS '순번';
COMMENT ON COLUMN mimiciv_hosp.omr.result_name IS '측정 항목명 (혈압, 체중 등)';
COMMENT ON COLUMN mimiciv_hosp.omr.result_value IS '측정값';

-- pharmacy
COMMENT ON TABLE mimiciv_hosp.pharmacy IS '약국 처방 정보';
COMMENT ON COLUMN mimiciv_hosp.pharmacy.subject_id IS '환자 고유 ID';
COMMENT ON COLUMN mimiciv_hosp.pharmacy.hadm_id IS '입원 고유 ID';
COMMENT ON COLUMN mimiciv_hosp.pharmacy.pharmacy_id IS '약국 처방 ID';
COMMENT ON COLUMN mimiciv_hosp.pharmacy.poe_id IS '처방 ID';
COMMENT ON COLUMN mimiciv_hosp.pharmacy.starttime IS '투약 시작 일시';
COMMENT ON COLUMN mimiciv_hosp.pharmacy.stoptime IS '투약 종료 일시';
COMMENT ON COLUMN mimiciv_hosp.pharmacy.medication IS '약물명';
COMMENT ON COLUMN mimiciv_hosp.pharmacy.proc_type IS '처방 유형';
COMMENT ON COLUMN mimiciv_hosp.pharmacy.status IS '처방 상태 (활성, 중단 등)';
COMMENT ON COLUMN mimiciv_hosp.pharmacy.entertime IS '입력 일시';
COMMENT ON COLUMN mimiciv_hosp.pharmacy.verifiedtime IS '약사 검증 일시';
COMMENT ON COLUMN mimiciv_hosp.pharmacy.route IS '투여 경로';
COMMENT ON COLUMN mimiciv_hosp.pharmacy.frequency IS '투여 빈도';
COMMENT ON COLUMN mimiciv_hosp.pharmacy.disp_sched IS '조제 일정';
COMMENT ON COLUMN mimiciv_hosp.pharmacy.infusion_type IS '주입 유형';
COMMENT ON COLUMN mimiciv_hosp.pharmacy.sliding_scale IS '슬라이딩 스케일 여부';
COMMENT ON COLUMN mimiciv_hosp.pharmacy.lockout_interval IS '잠금 간격';
COMMENT ON COLUMN mimiciv_hosp.pharmacy.basal_rate IS '기저 주입 속도';
COMMENT ON COLUMN mimiciv_hosp.pharmacy.one_hr_max IS '시간당 최대량';
COMMENT ON COLUMN mimiciv_hosp.pharmacy.doses_per_24_hrs IS '24시간 투여 횟수';
COMMENT ON COLUMN mimiciv_hosp.pharmacy.duration IS '투여 기간';
COMMENT ON COLUMN mimiciv_hosp.pharmacy.duration_interval IS '투여 기간 단위';
COMMENT ON COLUMN mimiciv_hosp.pharmacy.expiration_value IS '유효기간 값';
COMMENT ON COLUMN mimiciv_hosp.pharmacy.expiration_unit IS '유효기간 단위';
COMMENT ON COLUMN mimiciv_hosp.pharmacy.expirationdate IS '만료 일시';
COMMENT ON COLUMN mimiciv_hosp.pharmacy.dispensation IS '조제 방식';
COMMENT ON COLUMN mimiciv_hosp.pharmacy.fill_quantity IS '조제 수량';

-- poe
COMMENT ON TABLE mimiciv_hosp.poe IS '처방 입력 기록 (Provider Order Entry)';
COMMENT ON COLUMN mimiciv_hosp.poe.poe_id IS '처방 고유 ID';
COMMENT ON COLUMN mimiciv_hosp.poe.poe_seq IS '처방 순번';
COMMENT ON COLUMN mimiciv_hosp.poe.subject_id IS '환자 고유 ID';
COMMENT ON COLUMN mimiciv_hosp.poe.hadm_id IS '입원 고유 ID';
COMMENT ON COLUMN mimiciv_hosp.poe.ordertime IS '처방 일시';
COMMENT ON COLUMN mimiciv_hosp.poe.order_type IS '처방 유형 (약물, 검사 등)';
COMMENT ON COLUMN mimiciv_hosp.poe.order_subtype IS '처방 세부 유형';
COMMENT ON COLUMN mimiciv_hosp.poe.transaction_type IS '트랜잭션 유형 (신규, 변경, 취소)';
COMMENT ON COLUMN mimiciv_hosp.poe.discontinue_of_poe_id IS '중단 대상 처방 ID';
COMMENT ON COLUMN mimiciv_hosp.poe.discontinued_by_poe_id IS '중단 처방 ID';
COMMENT ON COLUMN mimiciv_hosp.poe.order_provider_id IS '처방 의료진 ID';
COMMENT ON COLUMN mimiciv_hosp.poe.order_status IS '처방 상태';

-- poe_detail
COMMENT ON TABLE mimiciv_hosp.poe_detail IS '처방 상세 정보';
COMMENT ON COLUMN mimiciv_hosp.poe_detail.poe_id IS '처방 고유 ID';
COMMENT ON COLUMN mimiciv_hosp.poe_detail.poe_seq IS '처방 순번';
COMMENT ON COLUMN mimiciv_hosp.poe_detail.subject_id IS '환자 고유 ID';
COMMENT ON COLUMN mimiciv_hosp.poe_detail.field_name IS '필드명';
COMMENT ON COLUMN mimiciv_hosp.poe_detail.field_value IS '필드값';

-- prescriptions
COMMENT ON TABLE mimiciv_hosp.prescriptions IS '처방전 (약물 처방 내역)';
COMMENT ON COLUMN mimiciv_hosp.prescriptions.subject_id IS '환자 고유 ID';
COMMENT ON COLUMN mimiciv_hosp.prescriptions.hadm_id IS '입원 고유 ID';
COMMENT ON COLUMN mimiciv_hosp.prescriptions.pharmacy_id IS '약국 처방 ID';
COMMENT ON COLUMN mimiciv_hosp.prescriptions.poe_id IS '처방 ID';
COMMENT ON COLUMN mimiciv_hosp.prescriptions.poe_seq IS '처방 순번';
COMMENT ON COLUMN mimiciv_hosp.prescriptions.order_provider_id IS '처방 의료진 ID';
COMMENT ON COLUMN mimiciv_hosp.prescriptions.starttime IS '투약 시작 일시';
COMMENT ON COLUMN mimiciv_hosp.prescriptions.stoptime IS '투약 종료 일시';
COMMENT ON COLUMN mimiciv_hosp.prescriptions.drug_type IS '약물 유형 (일반, 기본 등)';
COMMENT ON COLUMN mimiciv_hosp.prescriptions.drug IS '약물명';
COMMENT ON COLUMN mimiciv_hosp.prescriptions.formulary_drug_cd IS '처방집 약물 코드';
COMMENT ON COLUMN mimiciv_hosp.prescriptions.gsn IS 'GSN 코드 (Generic Sequence Number)';
COMMENT ON COLUMN mimiciv_hosp.prescriptions.ndc IS 'NDC 코드 (National Drug Code)';
COMMENT ON COLUMN mimiciv_hosp.prescriptions.prod_strength IS '제품 용량/농도';
COMMENT ON COLUMN mimiciv_hosp.prescriptions.form_rx IS '처방 제형';
COMMENT ON COLUMN mimiciv_hosp.prescriptions.dose_val_rx IS '처방 용량';
COMMENT ON COLUMN mimiciv_hosp.prescriptions.dose_unit_rx IS '처방 용량 단위';
COMMENT ON COLUMN mimiciv_hosp.prescriptions.form_val_disp IS '조제 제형 수량';
COMMENT ON COLUMN mimiciv_hosp.prescriptions.form_unit_disp IS '조제 제형 단위';
COMMENT ON COLUMN mimiciv_hosp.prescriptions.doses_per_24_hrs IS '24시간 투여 횟수';
COMMENT ON COLUMN mimiciv_hosp.prescriptions.route IS '투여 경로';

-- procedures_icd
COMMENT ON TABLE mimiciv_hosp.procedures_icd IS '환자별 ICD 시술/수술 기록';
COMMENT ON COLUMN mimiciv_hosp.procedures_icd.subject_id IS '환자 고유 ID';
COMMENT ON COLUMN mimiciv_hosp.procedures_icd.hadm_id IS '입원 고유 ID';
COMMENT ON COLUMN mimiciv_hosp.procedures_icd.seq_num IS '시술 순번';
COMMENT ON COLUMN mimiciv_hosp.procedures_icd.chartdate IS '시술 날짜';
COMMENT ON COLUMN mimiciv_hosp.procedures_icd.icd_code IS 'ICD 시술 코드';
COMMENT ON COLUMN mimiciv_hosp.procedures_icd.icd_version IS 'ICD 버전 (9 또는 10)';

-- provider
COMMENT ON TABLE mimiciv_hosp.provider IS '의료진 목록';
COMMENT ON COLUMN mimiciv_hosp.provider.provider_id IS '의료진 고유 ID';

-- services
COMMENT ON TABLE mimiciv_hosp.services IS '진료과 이동 기록';
COMMENT ON COLUMN mimiciv_hosp.services.subject_id IS '환자 고유 ID';
COMMENT ON COLUMN mimiciv_hosp.services.hadm_id IS '입원 고유 ID';
COMMENT ON COLUMN mimiciv_hosp.services.transfertime IS '이동 일시';
COMMENT ON COLUMN mimiciv_hosp.services.prev_service IS '이전 진료과';
COMMENT ON COLUMN mimiciv_hosp.services.curr_service IS '현재 진료과';

-- transfers
COMMENT ON TABLE mimiciv_hosp.transfers IS '병동/병실 이동 기록';
COMMENT ON COLUMN mimiciv_hosp.transfers.subject_id IS '환자 고유 ID';
COMMENT ON COLUMN mimiciv_hosp.transfers.hadm_id IS '입원 고유 ID';
COMMENT ON COLUMN mimiciv_hosp.transfers.transfer_id IS '이동 고유 ID';
COMMENT ON COLUMN mimiciv_hosp.transfers.eventtype IS '이벤트 유형 (입원, 이동, 퇴원)';
COMMENT ON COLUMN mimiciv_hosp.transfers.careunit IS '병동/부서명';
COMMENT ON COLUMN mimiciv_hosp.transfers.intime IS '입실 일시';
COMMENT ON COLUMN mimiciv_hosp.transfers.outtime IS '퇴실 일시';

-- =====================
-- mimiciv_icu 스키마
-- =====================

COMMENT ON SCHEMA mimiciv_icu IS 'ICU (중환자실) 정보';

-- caregiver
COMMENT ON TABLE mimiciv_icu.caregiver IS '간호 인력 목록';
COMMENT ON COLUMN mimiciv_icu.caregiver.caregiver_id IS '간호 인력 고유 ID';

-- chartevents
COMMENT ON TABLE mimiciv_icu.chartevents IS 'ICU 차트 기록 (활력징후, 간호기록 등)';
COMMENT ON COLUMN mimiciv_icu.chartevents.subject_id IS '환자 고유 ID';
COMMENT ON COLUMN mimiciv_icu.chartevents.hadm_id IS '입원 고유 ID';
COMMENT ON COLUMN mimiciv_icu.chartevents.stay_id IS 'ICU 입실 고유 ID';
COMMENT ON COLUMN mimiciv_icu.chartevents.caregiver_id IS '간호 인력 ID';
COMMENT ON COLUMN mimiciv_icu.chartevents.charttime IS '기록 일시';
COMMENT ON COLUMN mimiciv_icu.chartevents.storetime IS '시스템 저장 일시';
COMMENT ON COLUMN mimiciv_icu.chartevents.itemid IS '측정 항목 ID (d_items 참조)';
COMMENT ON COLUMN mimiciv_icu.chartevents.value IS '측정값 (문자열)';
COMMENT ON COLUMN mimiciv_icu.chartevents.valuenum IS '측정값 (숫자)';
COMMENT ON COLUMN mimiciv_icu.chartevents.valueuom IS '측정 단위';
COMMENT ON COLUMN mimiciv_icu.chartevents.warning IS '경고 여부';

-- d_items
COMMENT ON TABLE mimiciv_icu.d_items IS 'ICU 측정 항목 사전';
COMMENT ON COLUMN mimiciv_icu.d_items.itemid IS '항목 고유 ID';
COMMENT ON COLUMN mimiciv_icu.d_items.label IS '항목명';
COMMENT ON COLUMN mimiciv_icu.d_items.abbreviation IS '약어';
COMMENT ON COLUMN mimiciv_icu.d_items.linksto IS '연결 테이블명';
COMMENT ON COLUMN mimiciv_icu.d_items.category IS '분류';
COMMENT ON COLUMN mimiciv_icu.d_items.unitname IS '단위명';
COMMENT ON COLUMN mimiciv_icu.d_items.param_type IS '파라미터 유형';
COMMENT ON COLUMN mimiciv_icu.d_items.lownormalvalue IS '정상 범위 하한';
COMMENT ON COLUMN mimiciv_icu.d_items.highnormalvalue IS '정상 범위 상한';

-- datetimeevents
COMMENT ON TABLE mimiciv_icu.datetimeevents IS 'ICU 날짜/시간 이벤트 (체위변경, 튜브교체 등)';
COMMENT ON COLUMN mimiciv_icu.datetimeevents.subject_id IS '환자 고유 ID';
COMMENT ON COLUMN mimiciv_icu.datetimeevents.hadm_id IS '입원 고유 ID';
COMMENT ON COLUMN mimiciv_icu.datetimeevents.stay_id IS 'ICU 입실 고유 ID';
COMMENT ON COLUMN mimiciv_icu.datetimeevents.caregiver_id IS '간호 인력 ID';
COMMENT ON COLUMN mimiciv_icu.datetimeevents.charttime IS '기록 일시';
COMMENT ON COLUMN mimiciv_icu.datetimeevents.storetime IS '시스템 저장 일시';
COMMENT ON COLUMN mimiciv_icu.datetimeevents.itemid IS '항목 ID';
COMMENT ON COLUMN mimiciv_icu.datetimeevents.value IS '이벤트 일시값';
COMMENT ON COLUMN mimiciv_icu.datetimeevents.valueuom IS '단위';
COMMENT ON COLUMN mimiciv_icu.datetimeevents.warning IS '경고 여부';

-- icustays
COMMENT ON TABLE mimiciv_icu.icustays IS 'ICU 입퇴실 기록';
COMMENT ON COLUMN mimiciv_icu.icustays.subject_id IS '환자 고유 ID';
COMMENT ON COLUMN mimiciv_icu.icustays.hadm_id IS '입원 고유 ID';
COMMENT ON COLUMN mimiciv_icu.icustays.stay_id IS 'ICU 입실 고유 ID';
COMMENT ON COLUMN mimiciv_icu.icustays.first_careunit IS '최초 ICU 병동';
COMMENT ON COLUMN mimiciv_icu.icustays.last_careunit IS '최종 ICU 병동';
COMMENT ON COLUMN mimiciv_icu.icustays.intime IS 'ICU 입실 일시';
COMMENT ON COLUMN mimiciv_icu.icustays.outtime IS 'ICU 퇴실 일시';
COMMENT ON COLUMN mimiciv_icu.icustays.los IS 'ICU 재원일수 (일)';

-- ingredientevents
COMMENT ON TABLE mimiciv_icu.ingredientevents IS 'ICU 수액/약물 성분별 투입 기록';
COMMENT ON COLUMN mimiciv_icu.ingredientevents.subject_id IS '환자 고유 ID';
COMMENT ON COLUMN mimiciv_icu.ingredientevents.hadm_id IS '입원 고유 ID';
COMMENT ON COLUMN mimiciv_icu.ingredientevents.stay_id IS 'ICU 입실 고유 ID';
COMMENT ON COLUMN mimiciv_icu.ingredientevents.caregiver_id IS '간호 인력 ID';
COMMENT ON COLUMN mimiciv_icu.ingredientevents.starttime IS '투입 시작 일시';
COMMENT ON COLUMN mimiciv_icu.ingredientevents.endtime IS '투입 종료 일시';
COMMENT ON COLUMN mimiciv_icu.ingredientevents.storetime IS '시스템 저장 일시';
COMMENT ON COLUMN mimiciv_icu.ingredientevents.itemid IS '성분 항목 ID';
COMMENT ON COLUMN mimiciv_icu.ingredientevents.amount IS '투입량';
COMMENT ON COLUMN mimiciv_icu.ingredientevents.amountuom IS '투입량 단위';
COMMENT ON COLUMN mimiciv_icu.ingredientevents.rate IS '투입 속도';
COMMENT ON COLUMN mimiciv_icu.ingredientevents.rateuom IS '투입 속도 단위';
COMMENT ON COLUMN mimiciv_icu.ingredientevents.orderid IS '처방 ID';
COMMENT ON COLUMN mimiciv_icu.ingredientevents.linkorderid IS '연결 처방 ID';
COMMENT ON COLUMN mimiciv_icu.ingredientevents.statusdescription IS '상태 설명';
COMMENT ON COLUMN mimiciv_icu.ingredientevents.originalamount IS '원래 투입량';
COMMENT ON COLUMN mimiciv_icu.ingredientevents.originalrate IS '원래 투입 속도';

-- inputevents
COMMENT ON TABLE mimiciv_icu.inputevents IS 'ICU 수액/약물 투입 기록';
COMMENT ON COLUMN mimiciv_icu.inputevents.subject_id IS '환자 고유 ID';
COMMENT ON COLUMN mimiciv_icu.inputevents.hadm_id IS '입원 고유 ID';
COMMENT ON COLUMN mimiciv_icu.inputevents.stay_id IS 'ICU 입실 고유 ID';
COMMENT ON COLUMN mimiciv_icu.inputevents.caregiver_id IS '간호 인력 ID';
COMMENT ON COLUMN mimiciv_icu.inputevents.starttime IS '투입 시작 일시';
COMMENT ON COLUMN mimiciv_icu.inputevents.endtime IS '투입 종료 일시';
COMMENT ON COLUMN mimiciv_icu.inputevents.storetime IS '시스템 저장 일시';
COMMENT ON COLUMN mimiciv_icu.inputevents.itemid IS '항목 ID';
COMMENT ON COLUMN mimiciv_icu.inputevents.amount IS '투입량';
COMMENT ON COLUMN mimiciv_icu.inputevents.amountuom IS '투입량 단위';
COMMENT ON COLUMN mimiciv_icu.inputevents.rate IS '투입 속도';
COMMENT ON COLUMN mimiciv_icu.inputevents.rateuom IS '투입 속도 단위';
COMMENT ON COLUMN mimiciv_icu.inputevents.orderid IS '처방 ID';
COMMENT ON COLUMN mimiciv_icu.inputevents.linkorderid IS '연결 처방 ID';
COMMENT ON COLUMN mimiciv_icu.inputevents.ordercategoryname IS '처방 분류명';
COMMENT ON COLUMN mimiciv_icu.inputevents.secondaryordercategoryname IS '2차 처방 분류명';
COMMENT ON COLUMN mimiciv_icu.inputevents.ordercomponenttypedescription IS '처방 구성요소 유형';
COMMENT ON COLUMN mimiciv_icu.inputevents.ordercategorydescription IS '처방 분류 설명';
COMMENT ON COLUMN mimiciv_icu.inputevents.patientweight IS '환자 체중 (kg)';
COMMENT ON COLUMN mimiciv_icu.inputevents.totalamount IS '총 투입량';
COMMENT ON COLUMN mimiciv_icu.inputevents.totalamountuom IS '총 투입량 단위';
COMMENT ON COLUMN mimiciv_icu.inputevents.isopenbag IS '개봉 백 여부';
COMMENT ON COLUMN mimiciv_icu.inputevents.continueinnextdept IS '다음 부서 계속 투입 여부';
COMMENT ON COLUMN mimiciv_icu.inputevents.statusdescription IS '상태 설명';
COMMENT ON COLUMN mimiciv_icu.inputevents.originalamount IS '원래 투입량';
COMMENT ON COLUMN mimiciv_icu.inputevents.originalrate IS '원래 투입 속도';

-- outputevents
COMMENT ON TABLE mimiciv_icu.outputevents IS 'ICU 배출량 기록 (소변, 배액 등)';
COMMENT ON COLUMN mimiciv_icu.outputevents.subject_id IS '환자 고유 ID';
COMMENT ON COLUMN mimiciv_icu.outputevents.hadm_id IS '입원 고유 ID';
COMMENT ON COLUMN mimiciv_icu.outputevents.stay_id IS 'ICU 입실 고유 ID';
COMMENT ON COLUMN mimiciv_icu.outputevents.caregiver_id IS '간호 인력 ID';
COMMENT ON COLUMN mimiciv_icu.outputevents.charttime IS '기록 일시';
COMMENT ON COLUMN mimiciv_icu.outputevents.storetime IS '시스템 저장 일시';
COMMENT ON COLUMN mimiciv_icu.outputevents.itemid IS '항목 ID';
COMMENT ON COLUMN mimiciv_icu.outputevents.value IS '배출량';
COMMENT ON COLUMN mimiciv_icu.outputevents.valueuom IS '배출량 단위';

-- procedureevents
COMMENT ON TABLE mimiciv_icu.procedureevents IS 'ICU 시술/처치 기록';
COMMENT ON COLUMN mimiciv_icu.procedureevents.subject_id IS '환자 고유 ID';
COMMENT ON COLUMN mimiciv_icu.procedureevents.hadm_id IS '입원 고유 ID';
COMMENT ON COLUMN mimiciv_icu.procedureevents.stay_id IS 'ICU 입실 고유 ID';
COMMENT ON COLUMN mimiciv_icu.procedureevents.caregiver_id IS '간호 인력 ID';
COMMENT ON COLUMN mimiciv_icu.procedureevents.starttime IS '시술 시작 일시';
COMMENT ON COLUMN mimiciv_icu.procedureevents.endtime IS '시술 종료 일시';
COMMENT ON COLUMN mimiciv_icu.procedureevents.storetime IS '시스템 저장 일시';
COMMENT ON COLUMN mimiciv_icu.procedureevents.itemid IS '시술 항목 ID';
COMMENT ON COLUMN mimiciv_icu.procedureevents.value IS '시술 값';
COMMENT ON COLUMN mimiciv_icu.procedureevents.valueuom IS '시술 단위';
COMMENT ON COLUMN mimiciv_icu.procedureevents.location IS '시술 위치';
COMMENT ON COLUMN mimiciv_icu.procedureevents.locationcategory IS '시술 위치 분류';
COMMENT ON COLUMN mimiciv_icu.procedureevents.orderid IS '처방 ID';
COMMENT ON COLUMN mimiciv_icu.procedureevents.linkorderid IS '연결 처방 ID';
COMMENT ON COLUMN mimiciv_icu.procedureevents.ordercategoryname IS '처방 분류명';
COMMENT ON COLUMN mimiciv_icu.procedureevents.ordercategorydescription IS '처방 분류 설명';
COMMENT ON COLUMN mimiciv_icu.procedureevents.patientweight IS '환자 체중 (kg)';
COMMENT ON COLUMN mimiciv_icu.procedureevents.isopenbag IS '개봉 백 여부';
COMMENT ON COLUMN mimiciv_icu.procedureevents.continueinnextdept IS '다음 부서 계속 여부';
COMMENT ON COLUMN mimiciv_icu.procedureevents.statusdescription IS '상태 설명';
COMMENT ON COLUMN mimiciv_icu.procedureevents.originalamount IS '원래 양';
COMMENT ON COLUMN mimiciv_icu.procedureevents.originalrate IS '원래 속도';

-- =====================
-- mimiciv_ecg 스키마
-- =====================

COMMENT ON SCHEMA mimiciv_ecg IS 'ECG (심전도) 파형 및 기계 판독 데이터';

-- record_list
COMMENT ON TABLE mimiciv_ecg.record_list IS 'ECG 기록 목록 (파형 파일 경로 포함)';
COMMENT ON COLUMN mimiciv_ecg.record_list.subject_id IS '환자 고유 ID';
COMMENT ON COLUMN mimiciv_ecg.record_list.study_id IS 'ECG 검사 고유 ID';
COMMENT ON COLUMN mimiciv_ecg.record_list.file_name IS '파일명';
COMMENT ON COLUMN mimiciv_ecg.record_list.ecg_time IS 'ECG 측정 일시';
COMMENT ON COLUMN mimiciv_ecg.record_list.path IS '파형 파일 경로 (.dat/.hea)';

-- machine_measurements
COMMENT ON TABLE mimiciv_ecg.machine_measurements IS 'ECG 기계 자동 판독 결과 (진단 + 측정값)';
COMMENT ON COLUMN mimiciv_ecg.machine_measurements.subject_id IS '환자 고유 ID';
COMMENT ON COLUMN mimiciv_ecg.machine_measurements.study_id IS 'ECG 검사 고유 ID';
COMMENT ON COLUMN mimiciv_ecg.machine_measurements.cart_id IS 'ECG 장비 ID';
COMMENT ON COLUMN mimiciv_ecg.machine_measurements.ecg_time IS 'ECG 측정 일시';
COMMENT ON COLUMN mimiciv_ecg.machine_measurements.report_0 IS '기계 판독 진단 1 (주진단)';
COMMENT ON COLUMN mimiciv_ecg.machine_measurements.report_1 IS '기계 판독 진단 2';
COMMENT ON COLUMN mimiciv_ecg.machine_measurements.report_2 IS '기계 판독 진단 3';
COMMENT ON COLUMN mimiciv_ecg.machine_measurements.report_3 IS '기계 판독 진단 4';
COMMENT ON COLUMN mimiciv_ecg.machine_measurements.report_4 IS '기계 판독 진단 5';
COMMENT ON COLUMN mimiciv_ecg.machine_measurements.report_5 IS '기계 판독 진단 6';
COMMENT ON COLUMN mimiciv_ecg.machine_measurements.report_6 IS '기계 판독 진단 7';
COMMENT ON COLUMN mimiciv_ecg.machine_measurements.report_7 IS '기계 판독 진단 8';
COMMENT ON COLUMN mimiciv_ecg.machine_measurements.report_8 IS '기계 판독 진단 9';
COMMENT ON COLUMN mimiciv_ecg.machine_measurements.report_9 IS '기계 판독 진단 10';
COMMENT ON COLUMN mimiciv_ecg.machine_measurements.report_10 IS '기계 판독 진단 11';
COMMENT ON COLUMN mimiciv_ecg.machine_measurements.report_11 IS '기계 판독 진단 12';
COMMENT ON COLUMN mimiciv_ecg.machine_measurements.report_12 IS '기계 판독 진단 13';
COMMENT ON COLUMN mimiciv_ecg.machine_measurements.report_13 IS '기계 판독 진단 14';
COMMENT ON COLUMN mimiciv_ecg.machine_measurements.report_14 IS '기계 판독 진단 15';
COMMENT ON COLUMN mimiciv_ecg.machine_measurements.report_15 IS '기계 판독 진단 16';
COMMENT ON COLUMN mimiciv_ecg.machine_measurements.report_16 IS '기계 판독 진단 17';
COMMENT ON COLUMN mimiciv_ecg.machine_measurements.report_17 IS '기계 판독 진단 18';
COMMENT ON COLUMN mimiciv_ecg.machine_measurements.bandwidth IS 'ECG 대역폭 설정';
COMMENT ON COLUMN mimiciv_ecg.machine_measurements.filtering IS '필터링 설정 (노치, 베이스라인)';
COMMENT ON COLUMN mimiciv_ecg.machine_measurements.rr_interval IS 'R-R 간격 (밀리초, 심박수 관련)';
COMMENT ON COLUMN mimiciv_ecg.machine_measurements.p_onset IS 'P파 시작점 (밀리초)';
COMMENT ON COLUMN mimiciv_ecg.machine_measurements.p_end IS 'P파 종료점 (밀리초)';
COMMENT ON COLUMN mimiciv_ecg.machine_measurements.qrs_onset IS 'QRS 복합파 시작점 (밀리초)';
COMMENT ON COLUMN mimiciv_ecg.machine_measurements.qrs_end IS 'QRS 복합파 종료점 (밀리초)';
COMMENT ON COLUMN mimiciv_ecg.machine_measurements.t_end IS 'T파 종료점 (밀리초)';
COMMENT ON COLUMN mimiciv_ecg.machine_measurements.p_axis IS 'P파 전기축 (도)';
COMMENT ON COLUMN mimiciv_ecg.machine_measurements.qrs_axis IS 'QRS 전기축 (도)';
COMMENT ON COLUMN mimiciv_ecg.machine_measurements.t_axis IS 'T파 전기축 (도)';

-- waveform_note_links
COMMENT ON TABLE mimiciv_ecg.waveform_note_links IS 'ECG 파형과 임상 노트 연결';
COMMENT ON COLUMN mimiciv_ecg.waveform_note_links.subject_id IS '환자 고유 ID';
COMMENT ON COLUMN mimiciv_ecg.waveform_note_links.study_id IS 'ECG 검사 고유 ID';
COMMENT ON COLUMN mimiciv_ecg.waveform_note_links.waveform_path IS '파형 파일 경로';
COMMENT ON COLUMN mimiciv_ecg.waveform_note_links.note_id IS '임상 노트 ID';
COMMENT ON COLUMN mimiciv_ecg.waveform_note_links.note_seq IS '노트 순번';
COMMENT ON COLUMN mimiciv_ecg.waveform_note_links.charttime IS '기록 일시';
