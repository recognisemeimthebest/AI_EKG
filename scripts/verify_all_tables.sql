-- Full data load verification
SELECT 'mimiciv_hosp.patients' as tbl, COUNT(*) as rows FROM mimiciv_hosp.patients
UNION ALL SELECT 'mimiciv_hosp.admissions', COUNT(*) FROM mimiciv_hosp.admissions
UNION ALL SELECT 'mimiciv_hosp.d_hcpcs', COUNT(*) FROM mimiciv_hosp.d_hcpcs
UNION ALL SELECT 'mimiciv_hosp.d_icd_diagnoses', COUNT(*) FROM mimiciv_hosp.d_icd_diagnoses
UNION ALL SELECT 'mimiciv_hosp.d_icd_procedures', COUNT(*) FROM mimiciv_hosp.d_icd_procedures
UNION ALL SELECT 'mimiciv_hosp.d_labitems', COUNT(*) FROM mimiciv_hosp.d_labitems
UNION ALL SELECT 'mimiciv_hosp.diagnoses_icd', COUNT(*) FROM mimiciv_hosp.diagnoses_icd
UNION ALL SELECT 'mimiciv_hosp.drgcodes', COUNT(*) FROM mimiciv_hosp.drgcodes
UNION ALL SELECT 'mimiciv_hosp.emar', COUNT(*) FROM mimiciv_hosp.emar
UNION ALL SELECT 'mimiciv_hosp.emar_detail', COUNT(*) FROM mimiciv_hosp.emar_detail
UNION ALL SELECT 'mimiciv_hosp.hcpcsevents', COUNT(*) FROM mimiciv_hosp.hcpcsevents
UNION ALL SELECT 'mimiciv_hosp.labevents', COUNT(*) FROM mimiciv_hosp.labevents
UNION ALL SELECT 'mimiciv_hosp.microbiologyevents', COUNT(*) FROM mimiciv_hosp.microbiologyevents
UNION ALL SELECT 'mimiciv_hosp.omr', COUNT(*) FROM mimiciv_hosp.omr
UNION ALL SELECT 'mimiciv_hosp.pharmacy', COUNT(*) FROM mimiciv_hosp.pharmacy
UNION ALL SELECT 'mimiciv_hosp.poe', COUNT(*) FROM mimiciv_hosp.poe
UNION ALL SELECT 'mimiciv_hosp.poe_detail', COUNT(*) FROM mimiciv_hosp.poe_detail
UNION ALL SELECT 'mimiciv_hosp.prescriptions', COUNT(*) FROM mimiciv_hosp.prescriptions
UNION ALL SELECT 'mimiciv_hosp.procedures_icd', COUNT(*) FROM mimiciv_hosp.procedures_icd
UNION ALL SELECT 'mimiciv_hosp.provider', COUNT(*) FROM mimiciv_hosp.provider
UNION ALL SELECT 'mimiciv_hosp.services', COUNT(*) FROM mimiciv_hosp.services
UNION ALL SELECT 'mimiciv_hosp.transfers', COUNT(*) FROM mimiciv_hosp.transfers
UNION ALL SELECT 'mimiciv_icu.caregiver', COUNT(*) FROM mimiciv_icu.caregiver
UNION ALL SELECT 'mimiciv_icu.chartevents', COUNT(*) FROM mimiciv_icu.chartevents
UNION ALL SELECT 'mimiciv_icu.d_items', COUNT(*) FROM mimiciv_icu.d_items
UNION ALL SELECT 'mimiciv_icu.datetimeevents', COUNT(*) FROM mimiciv_icu.datetimeevents
UNION ALL SELECT 'mimiciv_icu.icustays', COUNT(*) FROM mimiciv_icu.icustays
UNION ALL SELECT 'mimiciv_icu.ingredientevents', COUNT(*) FROM mimiciv_icu.ingredientevents
UNION ALL SELECT 'mimiciv_icu.inputevents', COUNT(*) FROM mimiciv_icu.inputevents
UNION ALL SELECT 'mimiciv_icu.outputevents', COUNT(*) FROM mimiciv_icu.outputevents
UNION ALL SELECT 'mimiciv_icu.procedureevents', COUNT(*) FROM mimiciv_icu.procedureevents
UNION ALL SELECT 'mimiciv_ecg.record_list', COUNT(*) FROM mimiciv_ecg.record_list
UNION ALL SELECT 'mimiciv_ecg.machine_measurements', COUNT(*) FROM mimiciv_ecg.machine_measurements
UNION ALL SELECT 'mimiciv_ecg.waveform_note_links', COUNT(*) FROM mimiciv_ecg.waveform_note_links
ORDER BY tbl;
