-- ECG + patient info matching
SELECT
    p.subject_id,
    p.gender,
    p.anchor_age,
    e.study_id,
    e.ecg_time,
    m.report_0 as diagnosis,
    m.report_1 as detail,
    m.rr_interval,
    m.qrs_axis,
    e.path as waveform_path
FROM mimiciv_ecg.record_list e
JOIN mimiciv_hosp.patients p ON p.subject_id = e.subject_id
JOIN mimiciv_ecg.machine_measurements m ON m.study_id = e.study_id
WHERE m.report_0 IS NOT NULL
LIMIT 10;

-- diagnosis distribution (top 20)
SELECT m.report_0 as diagnosis, COUNT(*) as cnt
FROM mimiciv_ecg.machine_measurements m
WHERE m.report_0 IS NOT NULL
GROUP BY m.report_0
ORDER BY cnt DESC
LIMIT 20;

-- total stats
SELECT
    (SELECT COUNT(*) FROM mimiciv_ecg.record_list) as total_ecg,
    (SELECT COUNT(DISTINCT subject_id) FROM mimiciv_ecg.record_list) as unique_patients,
    (SELECT COUNT(*) FROM mimiciv_ecg.machine_measurements WHERE report_0 IS NOT NULL) as with_diagnosis;
