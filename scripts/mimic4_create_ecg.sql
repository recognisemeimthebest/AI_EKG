DROP SCHEMA IF EXISTS mimiciv_ecg CASCADE;
CREATE SCHEMA mimiciv_ecg;

CREATE TABLE mimiciv_ecg.record_list (
    subject_id INTEGER NOT NULL,
    study_id INTEGER NOT NULL,
    file_name VARCHAR(20),
    ecg_time TIMESTAMP,
    path TEXT,
    CONSTRAINT pk_record_list PRIMARY KEY (study_id)
);

CREATE TABLE mimiciv_ecg.machine_measurements (
    subject_id INTEGER NOT NULL,
    study_id INTEGER NOT NULL,
    cart_id INTEGER,
    ecg_time TIMESTAMP,
    report_0 TEXT,
    report_1 TEXT,
    report_2 TEXT,
    report_3 TEXT,
    report_4 TEXT,
    report_5 TEXT,
    report_6 TEXT,
    report_7 TEXT,
    report_8 TEXT,
    report_9 TEXT,
    report_10 TEXT,
    report_11 TEXT,
    report_12 TEXT,
    report_13 TEXT,
    report_14 TEXT,
    report_15 TEXT,
    report_16 TEXT,
    report_17 TEXT,
    bandwidth VARCHAR(30),
    filtering VARCHAR(50),
    rr_interval INTEGER,
    p_onset INTEGER,
    p_end INTEGER,
    qrs_onset INTEGER,
    qrs_end INTEGER,
    t_end INTEGER,
    p_axis INTEGER,
    qrs_axis INTEGER,
    t_axis INTEGER
);

CREATE TABLE mimiciv_ecg.waveform_note_links (
    subject_id INTEGER NOT NULL,
    study_id INTEGER NOT NULL,
    waveform_path TEXT,
    note_id VARCHAR(30),
    note_seq INTEGER,
    charttime TIMESTAMP
);
