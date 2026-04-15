# AI-EKG

AI-powered 3-lead EKG measurement device — hardware to software, end-to-end.

## Overview

A personal/bedside EKG device built with ESP32 + ADS1292R, featuring AI-based cardiac analysis powered by models trained on MIMIC-IV ECG data.

### AI Features

| Feature | Status | Performance |
|---------|--------|-------------|
| Arrhythmia Classification (Normal/AFib/Other) | Done | 90.6% accuracy |
| Paroxysmal AF Detection | Done | AUROC 0.8240 |
| AFib Prediction (15-day sequence + clinical) | In Progress | — |

### Hardware

- **MCU**: ESP32-WROOM-32D
- **ADC**: ADS1292R (24-bit, 2-ch)
- **Display**: 5" SPI LCD (real-time waveform)
- **AI Server**: Raspberry Pi 5 (BLE connection)
- **Power**: USB 5V + Li-ion battery backup

## Project Structure

```
ml/
  preprocessing/   # ECG signal processing pipeline
  model/           # Model architectures & training scripts
docs/              # Project plan, reports, research notes
scripts/           # MIMIC-IV database setup (SQL/shell)
references/        # Component surveys & datasheets
```

## ML Models

- **ResNet34** — 12-lead arrhythmia classifier
- **CNN-TCN** — temporal convolution network for rhythm analysis
- **ECG-FM** — foundation model fine-tuning for paroxysmal AF detection

Trained on **MIMIC-IV ECG** (~775K records) with PostgreSQL-backed clinical features.

## Setup

### Prerequisites

- Python 3.10+
- PostgreSQL 16 (with MIMIC-IV data loaded)

### Environment Variables

```bash
export DB_PASSWORD="your_postgres_password"
# Optional (defaults shown):
# export DB_HOST="localhost"
# export DB_PORT="5432"
# export DB_NAME="mimic4"
# export DB_USER="postgres"
```

### Install

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install torch numpy scipy wfdb psycopg2-binary h5py scikit-learn
```

## Data

This project uses [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/) and [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/) datasets. PhysioNet credentialed access is required.

## License

This project is for research and educational purposes.
