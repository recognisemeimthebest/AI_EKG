-- MIMIC-IV 3.1 Schema for PostgreSQL
-- hosp schema
DROP SCHEMA IF EXISTS mimiciv_hosp CASCADE;
CREATE SCHEMA mimiciv_hosp;

-- icu schema
DROP SCHEMA IF EXISTS mimiciv_icu CASCADE;
CREATE SCHEMA mimiciv_icu;

-- =====================
-- HOSP TABLES
-- =====================

CREATE TABLE mimiciv_hosp.patients (
    subject_id INTEGER NOT NULL,
    gender VARCHAR(1) NOT NULL,
    anchor_age INTEGER NOT NULL,
    anchor_year INTEGER NOT NULL,
    anchor_year_group VARCHAR(20) NOT NULL,
    dod DATE,
    CONSTRAINT pk_patients PRIMARY KEY (subject_id)
);

CREATE TABLE mimiciv_hosp.admissions (
    subject_id INTEGER NOT NULL,
    hadm_id INTEGER NOT NULL,
    admittime TIMESTAMP NOT NULL,
    dischtime TIMESTAMP NOT NULL,
    deathtime TIMESTAMP,
    admission_type VARCHAR(50) NOT NULL,
    admit_provider_id VARCHAR(10),
    admission_location VARCHAR(60),
    discharge_location VARCHAR(60),
    insurance VARCHAR(255),
    language VARCHAR(10),
    marital_status VARCHAR(30),
    race VARCHAR(80),
    edregtime TIMESTAMP,
    edouttime TIMESTAMP,
    hospital_expire_flag SMALLINT,
    CONSTRAINT pk_admissions PRIMARY KEY (hadm_id)
);

CREATE TABLE mimiciv_hosp.d_hcpcs (
    code VARCHAR(10) NOT NULL,
    category SMALLINT,
    long_description TEXT,
    short_description VARCHAR(180)
);

CREATE TABLE mimiciv_hosp.d_icd_diagnoses (
    icd_code VARCHAR(10) NOT NULL,
    icd_version SMALLINT NOT NULL,
    long_title VARCHAR(300) NOT NULL
);

CREATE TABLE mimiciv_hosp.d_icd_procedures (
    icd_code VARCHAR(10) NOT NULL,
    icd_version SMALLINT NOT NULL,
    long_title VARCHAR(300) NOT NULL
);

CREATE TABLE mimiciv_hosp.d_labitems (
    itemid INTEGER NOT NULL,
    label VARCHAR(50),
    fluid VARCHAR(50),
    category VARCHAR(50),
    CONSTRAINT pk_d_labitems PRIMARY KEY (itemid)
);

CREATE TABLE mimiciv_hosp.diagnoses_icd (
    subject_id INTEGER NOT NULL,
    hadm_id INTEGER NOT NULL,
    seq_num INTEGER NOT NULL,
    icd_code VARCHAR(10) NOT NULL,
    icd_version SMALLINT NOT NULL
);

CREATE TABLE mimiciv_hosp.drgcodes (
    subject_id INTEGER NOT NULL,
    hadm_id INTEGER NOT NULL,
    drg_type VARCHAR(4) NOT NULL,
    drg_code VARCHAR(10) NOT NULL,
    description VARCHAR(300),
    drg_severity SMALLINT,
    drg_mortality SMALLINT
);

CREATE TABLE mimiciv_hosp.emar (
    subject_id INTEGER NOT NULL,
    hadm_id INTEGER,
    emar_id VARCHAR(25) NOT NULL,
    emar_seq INTEGER NOT NULL,
    poe_id VARCHAR(25),
    pharmacy_id INTEGER,
    enter_provider_id VARCHAR(10),
    charttime TIMESTAMP NOT NULL,
    medication TEXT,
    event_txt VARCHAR(100),
    scheduletime TIMESTAMP,
    storetime TIMESTAMP NOT NULL
);

CREATE TABLE mimiciv_hosp.emar_detail (
    subject_id INTEGER NOT NULL,
    emar_id VARCHAR(25) NOT NULL,
    emar_seq INTEGER NOT NULL,
    parent_field_ordinal VARCHAR(10),
    administration_type VARCHAR(50),
    pharmacy_id INTEGER,
    barcode_type VARCHAR(4),
    reason_for_no_barcode TEXT,
    complete_dose_not_given VARCHAR(5),
    dose_due VARCHAR(100),
    dose_due_unit VARCHAR(50),
    dose_given VARCHAR(255),
    dose_given_unit VARCHAR(50),
    will_remainder_of_dose_be_given VARCHAR(5),
    product_amount_given VARCHAR(30),
    product_unit VARCHAR(30),
    product_code VARCHAR(30),
    product_description VARCHAR(255),
    product_description_other VARCHAR(255),
    prior_infusion_rate VARCHAR(20),
    infusion_rate VARCHAR(20),
    infusion_rate_adjustment VARCHAR(50),
    infusion_rate_adjustment_amount VARCHAR(30),
    infusion_rate_unit VARCHAR(30),
    route VARCHAR(30),
    infusion_complete VARCHAR(1),
    completion_interval VARCHAR(30),
    new_iv_bag_hung VARCHAR(1),
    continued_infusion_in_other_location VARCHAR(1),
    restart_interval VARCHAR(50),
    side VARCHAR(10),
    site VARCHAR(255),
    non_formulary_visual_verification VARCHAR(1)
);

CREATE TABLE mimiciv_hosp.hcpcsevents (
    subject_id INTEGER NOT NULL,
    hadm_id INTEGER NOT NULL,
    chartdate DATE NOT NULL,
    hcpcs_cd VARCHAR(10) NOT NULL,
    seq_num INTEGER NOT NULL,
    short_description VARCHAR(180)
);

CREATE TABLE mimiciv_hosp.labevents (
    labevent_id INTEGER NOT NULL,
    subject_id INTEGER NOT NULL,
    hadm_id INTEGER,
    specimen_id INTEGER NOT NULL,
    itemid INTEGER NOT NULL,
    order_provider_id VARCHAR(10),
    charttime TIMESTAMP NOT NULL,
    storetime TIMESTAMP,
    value VARCHAR(200),
    valuenum DOUBLE PRECISION,
    valueuom VARCHAR(20),
    ref_range_lower DOUBLE PRECISION,
    ref_range_upper DOUBLE PRECISION,
    flag VARCHAR(10),
    priority VARCHAR(7),
    comments TEXT
);

CREATE TABLE mimiciv_hosp.microbiologyevents (
    microevent_id INTEGER NOT NULL,
    subject_id INTEGER NOT NULL,
    hadm_id INTEGER,
    micro_specimen_id INTEGER NOT NULL,
    order_provider_id VARCHAR(10),
    chartdate DATE NOT NULL,
    charttime TIMESTAMP,
    spec_itemid INTEGER NOT NULL,
    spec_type_desc VARCHAR(100) NOT NULL,
    test_seq INTEGER NOT NULL,
    storedate DATE,
    storetime TIMESTAMP,
    test_itemid INTEGER NOT NULL,
    test_name VARCHAR(100) NOT NULL,
    org_itemid INTEGER,
    org_name VARCHAR(100),
    isolate_num SMALLINT,
    quantity VARCHAR(50),
    ab_itemid INTEGER,
    ab_name VARCHAR(30),
    dilution_text VARCHAR(10),
    dilution_comparison VARCHAR(20),
    dilution_value DOUBLE PRECISION,
    interpretation VARCHAR(5),
    comments TEXT
);

CREATE TABLE mimiciv_hosp.omr (
    subject_id INTEGER NOT NULL,
    chartdate DATE NOT NULL,
    seq_num INTEGER NOT NULL,
    result_name VARCHAR(100) NOT NULL,
    result_value TEXT
);

CREATE TABLE mimiciv_hosp.pharmacy (
    subject_id INTEGER NOT NULL,
    hadm_id INTEGER NOT NULL,
    pharmacy_id INTEGER NOT NULL,
    poe_id VARCHAR(25),
    starttime TIMESTAMP,
    stoptime TIMESTAMP,
    medication TEXT,
    proc_type VARCHAR(50),
    status VARCHAR(30),
    entertime TIMESTAMP,
    verifiedtime TIMESTAMP,
    route VARCHAR(30),
    frequency VARCHAR(30),
    disp_sched VARCHAR(100),
    infusion_type VARCHAR(15),
    sliding_scale VARCHAR(1),
    lockout_interval VARCHAR(50),
    basal_rate DOUBLE PRECISION,
    one_hr_max VARCHAR(10),
    doses_per_24_hrs DOUBLE PRECISION,
    duration DOUBLE PRECISION,
    duration_interval VARCHAR(50),
    expiration_value INTEGER,
    expiration_unit VARCHAR(50),
    expirationdate TIMESTAMP,
    dispensation VARCHAR(50),
    fill_quantity VARCHAR(30)
);

CREATE TABLE mimiciv_hosp.poe (
    poe_id VARCHAR(25) NOT NULL,
    poe_seq INTEGER NOT NULL,
    subject_id INTEGER NOT NULL,
    hadm_id INTEGER,
    ordertime TIMESTAMP NOT NULL,
    order_type VARCHAR(25) NOT NULL,
    order_subtype VARCHAR(50),
    transaction_type VARCHAR(15),
    discontinue_of_poe_id VARCHAR(25),
    discontinued_by_poe_id VARCHAR(25),
    order_provider_id VARCHAR(10),
    order_status VARCHAR(15)
);

CREATE TABLE mimiciv_hosp.poe_detail (
    poe_id VARCHAR(25) NOT NULL,
    poe_seq INTEGER NOT NULL,
    subject_id INTEGER NOT NULL,
    field_name VARCHAR(50) NOT NULL,
    field_value TEXT
);

CREATE TABLE mimiciv_hosp.prescriptions (
    subject_id INTEGER NOT NULL,
    hadm_id INTEGER NOT NULL,
    pharmacy_id INTEGER NOT NULL,
    poe_id VARCHAR(25),
    poe_seq INTEGER,
    order_provider_id VARCHAR(10),
    starttime TIMESTAMP,
    stoptime TIMESTAMP,
    drug_type VARCHAR(20),
    drug VARCHAR(255),
    formulary_drug_cd VARCHAR(50),
    gsn VARCHAR(200),
    ndc VARCHAR(25),
    prod_strength VARCHAR(120),
    form_rx VARCHAR(25),
    dose_val_rx VARCHAR(100),
    dose_unit_rx VARCHAR(50),
    form_val_disp VARCHAR(50),
    form_unit_disp VARCHAR(50),
    doses_per_24_hrs DOUBLE PRECISION,
    route VARCHAR(30)
);

CREATE TABLE mimiciv_hosp.procedures_icd (
    subject_id INTEGER NOT NULL,
    hadm_id INTEGER NOT NULL,
    seq_num INTEGER NOT NULL,
    chartdate DATE NOT NULL,
    icd_code VARCHAR(10) NOT NULL,
    icd_version SMALLINT NOT NULL
);

CREATE TABLE mimiciv_hosp.provider (
    provider_id VARCHAR(10) NOT NULL
);

CREATE TABLE mimiciv_hosp.services (
    subject_id INTEGER NOT NULL,
    hadm_id INTEGER NOT NULL,
    transfertime TIMESTAMP NOT NULL,
    prev_service VARCHAR(20),
    curr_service VARCHAR(20) NOT NULL
);

CREATE TABLE mimiciv_hosp.transfers (
    subject_id INTEGER NOT NULL,
    hadm_id INTEGER,
    transfer_id INTEGER NOT NULL,
    eventtype VARCHAR(10) NOT NULL,
    careunit VARCHAR(50),
    intime TIMESTAMP NOT NULL,
    outtime TIMESTAMP
);

-- =====================
-- ICU TABLES
-- =====================

CREATE TABLE mimiciv_icu.caregiver (
    caregiver_id INTEGER NOT NULL,
    CONSTRAINT pk_caregiver PRIMARY KEY (caregiver_id)
);

CREATE TABLE mimiciv_icu.chartevents (
    subject_id INTEGER NOT NULL,
    hadm_id INTEGER NOT NULL,
    stay_id INTEGER NOT NULL,
    caregiver_id INTEGER,
    charttime TIMESTAMP NOT NULL,
    storetime TIMESTAMP,
    itemid INTEGER NOT NULL,
    value VARCHAR(200),
    valuenum DOUBLE PRECISION,
    valueuom VARCHAR(20),
    warning SMALLINT
);

CREATE TABLE mimiciv_icu.d_items (
    itemid INTEGER NOT NULL,
    label VARCHAR(100) NOT NULL,
    abbreviation VARCHAR(50),
    linksto VARCHAR(30),
    category VARCHAR(50),
    unitname VARCHAR(50),
    param_type VARCHAR(30),
    lownormalvalue DOUBLE PRECISION,
    highnormalvalue DOUBLE PRECISION,
    CONSTRAINT pk_d_items PRIMARY KEY (itemid)
);

CREATE TABLE mimiciv_icu.datetimeevents (
    subject_id INTEGER NOT NULL,
    hadm_id INTEGER NOT NULL,
    stay_id INTEGER NOT NULL,
    caregiver_id INTEGER,
    charttime TIMESTAMP NOT NULL,
    storetime TIMESTAMP NOT NULL,
    itemid INTEGER NOT NULL,
    value TIMESTAMP NOT NULL,
    valueuom VARCHAR(20),
    warning SMALLINT
);

CREATE TABLE mimiciv_icu.icustays (
    subject_id INTEGER NOT NULL,
    hadm_id INTEGER NOT NULL,
    stay_id INTEGER NOT NULL,
    first_careunit VARCHAR(50),
    last_careunit VARCHAR(50),
    intime TIMESTAMP NOT NULL,
    outtime TIMESTAMP NOT NULL,
    los DOUBLE PRECISION,
    CONSTRAINT pk_icustays PRIMARY KEY (stay_id)
);

CREATE TABLE mimiciv_icu.ingredientevents (
    subject_id INTEGER NOT NULL,
    hadm_id INTEGER NOT NULL,
    stay_id INTEGER NOT NULL,
    caregiver_id INTEGER,
    starttime TIMESTAMP NOT NULL,
    endtime TIMESTAMP NOT NULL,
    storetime TIMESTAMP NOT NULL,
    itemid INTEGER NOT NULL,
    amount DOUBLE PRECISION,
    amountuom VARCHAR(20),
    rate DOUBLE PRECISION,
    rateuom VARCHAR(20),
    orderid INTEGER NOT NULL,
    linkorderid INTEGER,
    statusdescription VARCHAR(30),
    originalamount DOUBLE PRECISION,
    originalrate DOUBLE PRECISION
);

CREATE TABLE mimiciv_icu.inputevents (
    subject_id INTEGER NOT NULL,
    hadm_id INTEGER NOT NULL,
    stay_id INTEGER NOT NULL,
    caregiver_id INTEGER,
    starttime TIMESTAMP NOT NULL,
    endtime TIMESTAMP NOT NULL,
    storetime TIMESTAMP NOT NULL,
    itemid INTEGER NOT NULL,
    amount DOUBLE PRECISION,
    amountuom VARCHAR(20),
    rate DOUBLE PRECISION,
    rateuom VARCHAR(20),
    orderid INTEGER NOT NULL,
    linkorderid INTEGER,
    ordercategoryname VARCHAR(50),
    secondaryordercategoryname VARCHAR(50),
    ordercomponenttypedescription VARCHAR(100),
    ordercategorydescription VARCHAR(30),
    patientweight DOUBLE PRECISION,
    totalamount DOUBLE PRECISION,
    totalamountuom VARCHAR(20),
    isopenbag SMALLINT,
    continueinnextdept SMALLINT,
    statusdescription VARCHAR(30),
    originalamount DOUBLE PRECISION,
    originalrate DOUBLE PRECISION
);

CREATE TABLE mimiciv_icu.outputevents (
    subject_id INTEGER NOT NULL,
    hadm_id INTEGER NOT NULL,
    stay_id INTEGER NOT NULL,
    caregiver_id INTEGER,
    charttime TIMESTAMP NOT NULL,
    storetime TIMESTAMP NOT NULL,
    itemid INTEGER NOT NULL,
    value DOUBLE PRECISION,
    valueuom VARCHAR(20)
);

CREATE TABLE mimiciv_icu.procedureevents (
    subject_id INTEGER NOT NULL,
    hadm_id INTEGER NOT NULL,
    stay_id INTEGER NOT NULL,
    caregiver_id INTEGER,
    starttime TIMESTAMP NOT NULL,
    endtime TIMESTAMP NOT NULL,
    storetime TIMESTAMP NOT NULL,
    itemid INTEGER NOT NULL,
    value DOUBLE PRECISION,
    valueuom VARCHAR(20),
    location VARCHAR(50),
    locationcategory VARCHAR(50),
    orderid INTEGER NOT NULL,
    linkorderid INTEGER,
    ordercategoryname VARCHAR(50),
    ordercategorydescription VARCHAR(30),
    patientweight DOUBLE PRECISION,
    isopenbag SMALLINT,
    continueinnextdept SMALLINT,
    statusdescription VARCHAR(30),
    originalamount DOUBLE PRECISION,
    originalrate DOUBLE PRECISION
);
