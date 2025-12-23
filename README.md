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
├── DDOS_ML_Detection/              # DDoS detection models using XGBoost and PyTorch
└── Anamolous_Inbound_Traffic_Logs/ # UNDER DEVELOPMENT: Tailored detection using synthetic data
```

## Getting Started

1. Clone the repository:
```bash
   git clone https://github.com/Dannychz7/Cyber-ML-Detections.git
   cd Cyber-ML-Detections
```

## Dataset

**Recommended Dataset: CIC-IDS 2017**

- URL: https://www.unb.ca/cic/datasets/ids-2017.html
- Provides comprehensive, realistic network traffic with various attack types including botnets, DDoS attacks, and brute-force intrusions
- Includes essential network flow features: Flow Bytes/s, Source IP, Destination IP, Protocol, and Flow Duration
- Ideal for developing and evaluating inbound/outbound traffic volume anomaly detection solutions

**Disclaimer:** I do not claim ownership of the CIC-IDS 2017 dataset. All intellectual property rights are reserved by its original creators.

## Status

**Note:** Testing and pipeline functionality are still under refinement and not fully operational at this time.

## Contact

Maintained by @Dannychz7

---

*Future enhancements may include screenshots, additional datasets, and performance metrics.*