# Halo-CME-detection
# Real-Time CME Shock Warning System Using Aditya-L1 Data

This repository presents a real-time detection system for Coronal Mass Ejection (CME)-driven shocks using particle and plasma observations from Aditya-L1. The project integrates physics-informed statistical logic and deep learning to deliver early alerts and real time detection for halo CME impacts at L1.

## Overview

The system is composed of two independent yet complementary features:

### Feature 1: Statistical Anomaly Detector (STEPS-based)

A physics-informed early warning mechanism that monitors energetic particle data from the STEPS instrument. It includes:
- Flux anomaly detection using a rolling 12-hour baseline and dynamic IQR-based thresholding
- Persistence logic and high-energy band monitoring
- Validation via Cross-Entropy between pitch-angle bin distributions
- Confidence estimation and cooldown logic to avoid repetitive alerts

### Feature 2: Deep Learning-Based Detector (SWIS + MAG)

A CNN-LSTM-Attention model trained on solar wind and magnetic field data from SWIS and MAG instruments. The pipeline includes:
- End-to-end preprocessing and harmonization of particle, field, and entropy features
- Physics-aware feature engineering (Alfvén speed, plasma beta, Mach numbers, entropy)
- Automatic RH-confirmed shock labeling using Mahalanobis distance and compression ratios
- PVSS(Physics validated shock score) scoring with Monte Carlo dropout for uncertainty estimation

---

## Directory Structure

├── notebooks/
│ ├── Feature1_Warning_System.ipynb
│ └── Feature2_Real_Time_Shock_Detection.ipynb
│
├── src1/ # Feature 1 (STEPS) source code
│ ├── config.py
│ ├── warning_system.py
│ └── cross_entropy.py
│
├── src2/ # Feature 2 (ML) source code
│ ├── config.py
│ ├── data_preparation.py
│ ├── feature_engineering.py
│ └── shock_labeler.py
│
├── docs/ # Submission materials
│ ├── Team Interstellar one-pager.pdf
│ └── Bharatiya Antariksh Hackathon 2025 Idea Submission.pdf
│
└── README.md


---

## Manual: How to Run

### Requirements
- Python 3.8+
- Suggested environment: Google Colab or local Jupyter Notebook

Download data from here:- https://drive.google.com/drive/folders/1KP682x8tB9-upPKgjz5IJH3mVh-ns4ey

### Running Feature 1 (STEPS-based Statistical Detector)

1. Open `Feature1_Statistical_Shock_Detector.ipynb` from the `notebooks/` directory.
2. Upload your aligned STEPS + SWIS CSV file.
3. Run all cells to simulate particle anomaly detection.
4. Output: `issued_alerts.csv` file with warnings and confidence metrics.

### Running Feature 2 (ML-Based Deep Learning Detector)

1. Open `Feature2_CNNLSTM_Attention_Pipeline.ipynb`.
2. Upload:
   - Training file: `aditya_l1_master_training_data_5sec.parquet`
   - Test file: `swis_mag_master_combined.csv`
3. Run all cells to:
   - Engineer physics-informed features
   - Automatically label shocks via RH conditions
   - Train CNN-LSTM-Attention model
   - Generate PVSS predictions with uncertainty
4. Output: Top PVSS candidates, entropy analytics, and final diagnostic plots.

---

## Output Highlights

- `issued_alerts.csv`: Alerts with timestamps, anomaly scores, high-energy flux, and entropy validation
- PVSS score plot: Temporal evolution of shock probability with uncertainty bands
- Physics overlays: Plasma speed, magnetic field strength, entropy gradients

---

## Contributors

- [Ridhima Gupta](https://github.com/guptaridhima06)
- [Amey Taksali](https://github.com/CIPHERclux)
- [Nihira Patwardhan](https://github.com/Nihira8006)

---

## Submission Artifacts

- `docs/Team Interstellar one-pager.pdf`: Project summary one-pager for Bharatiya Antariksh Hackathon 2025
- `docs/Bharatiya Antariksh Hackathon 2025 Idea Submission.pdf`: Final submitted technical presentation

---

## License

This repository is released under the MIT License. You are free to use, modify, and distribute the code with attribution.



