cat << 'EOF' > README.md
# README: MOO-based PAT Framework for Protein Refolding

This repository contains the implementation of a process analytical technology (PAT) framework for real-time monitoring and autonomous control of protein refolding[cite: 19]. The system integrates a 1D-CNN-powered Fourier transform infrared (FTIR) soft sensor with a multi-objective optimization (MOO) scheme utilizing NSGA-II[cite: 20, 21].

---

## Project Overview

Protein refolding is typically a lengthy and inefficient bottleneck in bioprocessing[cite: 18]. This project proposes a real-time monitoring and control scheme to identify parameter deviations and autonomously adjust process settings, ensuring final yields remain within target limits[cite: 19, 23].

### Key Components
* **Soft Sensor:** A 1D-CNN model coupled with FTIR spectroscopy that tracks refolding dynamics with an average prediction error of ~5%[cite: 284, 1027].
* **Real-time Monitoring:** Continuous acquisition of Critical Process Parameters (CPPs) including pH, temperature, and Oxidation-Reduction Potential (ORP)[cite: 391, 526].
* **Autonomous Control (Scheme B):** An ensemble-coupled NSGA-II optimizer that provides real-time adjustments by identifying optimal conditions on the Pareto front[cite: 509, 1020].
* **Fault Diagnosis:** Demonstrated capability to identify non-viable "failure batches" (e.g., yield dropping to 10%) for early process termination[cite: 1028, 1029].

---

## Technical Specifications

### Hardware Environment
* **CPU:** Intel(R) Core™ i7-14700K (3.40 GHz) [cite: 141]
* **RAM:** 64 GB [cite: 141]
* **GPU:** NVIDIA T400 4 GB [cite: 141]

### 1D-CNN Architecture
* **Preprocessing:** PCA denoising (11 components), Savitzky-Golay filtering, Gaussian smoothing, and SNV baseline correction[cite: 178, 202].
* **Model Structure:** Four 1D-convolutional layers (32 filters, kernel sizes 8-16) followed by Batch Normalization, ReLU activation, and a 128-unit fully connected layer[cite: 248, 251, 261].
* **Performance:** Average inference latency of 10-15 milliseconds[cite: 303].

---

## Getting Started

### Prerequisites
* **Python Version:** 3.9.7 [cite: 165]
* **Primary Libraries:** NumPy, SciPy, scikit-learn, Keras, and TensorFlow [cite: 166]

### Installation
```bash
git clone [https://github.com/Naveen-567/Insulin-refolding-optimization](https://github.com/Naveen-567/Insulin-refolding-optimization)
cd Insulin-refolding-optimization
pip install numpy scipy scikit-learn tensorflow keras pandas matplotlib

### Usage
Calibration: Train the 1D-CNN model using the 200 datasets and spectral data provided in 1D-CNN.xlsx.
Setup: Configure immersion probes (pH, ORP, temperature) for data acquisition.
Execution: Use the Python middleware to trigger control actions via the NSGA-II optimizer when deviations occur.

### Citation
Jesubalan, N. G., Sharma, R., & Rathore, A. S. (2026). MOO-based implementation of process analytical technology framework: Protein refolding as a case study. AIChE Journal, e70335. doi:10.1002/aic.70335
