-- ECG FK constraints
ALTER TABLE mimiciv_ecg.record_list
ADD CONSTRAINT record_list_patients_fk
  FOREIGN KEY (subject_id) REFERENCES mimiciv_hosp.patients(subject_id);

ALTER TABLE mimiciv_ecg.machine_measurements
ADD CONSTRAINT machine_measurements_record_fk
  FOREIGN KEY (study_id) REFERENCES mimiciv_ecg.record_list(study_id);

ALTER TABLE mimiciv_ecg.waveform_note_links
ADD CONSTRAINT waveform_note_links_record_fk
  FOREIGN KEY (study_id) REFERENCES mimiciv_ecg.record_list(study_id);

-- ECG indexes
CREATE INDEX idx_record_list_subject_id ON mimiciv_ecg.record_list(subject_id);
CREATE INDEX idx_record_list_ecg_time ON mimiciv_ecg.record_list(ecg_time);
CREATE INDEX idx_machine_measurements_study_id ON mimiciv_ecg.machine_measurements(study_id);
CREATE INDEX idx_machine_measurements_subject_id ON mimiciv_ecg.machine_measurements(subject_id);
CREATE INDEX idx_waveform_note_links_study_id ON mimiciv_ecg.waveform_note_links(study_id);
CREATE INDEX idx_waveform_note_links_subject_id ON mimiciv_ecg.waveform_note_links(subject_id);
