# Cyber-ML-Detections

Welcome to my repository for proof-of-concept Machine Learning detection tools focused on cyber threat detection.

## Project Overview

This repository contains various machine learning models and scripts for detecting cyber threats using real network traffic datasets. The goal is to explore the feasibility and effectiveness of ML-based detection methods.

## Features

- XGBoost and PyTorch implementations
- Network traffic feature extraction and preprocessing
- Sample prediction pipelines
- Custom model training and evaluation

## Repository Structure
```
Cyber-ML-Detections/
├── DDOS_ML_Detection/              # DDoS detection models using XGBoost and PyTorch (OPERATIONAL)
└── Anamolous_Inbound_Traffic_Logs/ # UNDER DEVELOPMENT: Tailored detection using synthetic data
```

## Getting Started

1. Clone the repository:
```bash
   git clone https://github.com/Dannychz7/Cyber-ML-Detections.git
   cd Cyber-ML-Detections
```

2. Navigate to the DDoS detection directory:
```bash
   cd DDOS_ML_Detection
```

3. See the [DDOS_ML_Detection README](DDOS_ML_Detection/README.md) for detailed setup and usage instructions.

## Dataset

**Recommended Dataset: CIC-IDS 2017**

- URL: https://www.unb.ca/cic/datasets/ids-2017.html
- Provides comprehensive, realistic network traffic with various attack types including botnets, DDoS attacks, and brute-force intrusions
- Includes essential network flow features: Flow Bytes/s, Source IP, Destination IP, Protocol, and Flow Duration
- Ideal for developing and evaluating inbound/outbound traffic volume anomaly detection solutions

**Disclaimer:** I do not claim ownership of the CIC-IDS 2017 dataset. All intellectual property rights are reserved by its original creators.

## Project Status

### Operational
- **DDOS_ML_Detection** - Both PyTorch and XGBoost implementations are fully functional
  - PyTorch model training and prediction pipeline
  - XGBoost model training and prediction pipeline
  - Automated visualization and reporting
  - Sample data prediction capabilities

### Under Development
- **Anamolous_Inbound_Traffic_Logs** - Tailored detection using synthetic data (in progress)

## Quick Start - DDoS Detection

Both models are ready to use:

**PyTorch Model:**
```bash
cd DDOS_ML_Detection
python3 PTorch_main.py
```

**XGBoost Model:**
```bash
cd DDOS_ML_Detection
python3 XGBoost_main.py
```

For detailed instructions, model outputs, and prediction examples, see the [DDOS_ML_Detection README](DDOS_ML_Detection/README.md).

## Contact

Maintained by @Dannychz7

---

*Future enhancements may include additional detection models, expanded datasets, and enhanced performance metrics.*