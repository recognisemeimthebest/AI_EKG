#!/bin/bash
# MIMIC-IV 3.1 Data Loader
PSQL="/c/Program Files/PostgreSQL/16/bin/psql.exe"
export PGPASSWORD=tlsghktk6
DB="mimic4"
HOST="localhost"
USER="postgres"

DATA_DIR="g:/AIEKG/data/mimic-iv-3.1"

load_table() {
    local schema=$1
    local table=$2
    local file="$DATA_DIR/$schema/$table.csv.gz"

    if [ ! -f "$file" ]; then
        echo "SKIP: $file not found"
        return
    fi

    echo -n "Loading $schema.$table... "
    gunzip -c "$file" | "$PSQL" -U $USER -h $HOST -d $DB -c "\COPY mimiciv_${schema}.${table} FROM STDIN WITH (FORMAT csv, HEADER true, NULL '')" 2>&1
    echo "Done."
}

echo "========== HOSP =========="
load_table hosp patients
load_table hosp admissions
load_table hosp d_hcpcs
load_table hosp d_icd_diagnoses
load_table hosp d_icd_procedures
load_table hosp d_labitems
load_table hosp diagnoses_icd
load_table hosp drgcodes
load_table hosp emar
load_table hosp emar_detail
load_table hosp hcpcsevents
load_table hosp labevents
load_table hosp microbiologyevents
load_table hosp omr
load_table hosp pharmacy
load_table hosp poe
load_table hosp poe_detail
load_table hosp prescriptions
load_table hosp procedures_icd
load_table hosp provider
load_table hosp services
load_table hosp transfers

echo "========== ICU =========="
load_table icu caregiver
load_table icu chartevents
load_table icu d_items
load_table icu datetimeevents
load_table icu icustays
load_table icu ingredientevents
load_table icu inputevents
load_table icu outputevents
load_table icu procedureevents

echo "========== ALL DONE =========="
