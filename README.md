# FlowCast: Deep Learning-Based Short-Horizon Traffic Flow Forecasting

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/tests-23%20passed-brightgreen.svg)](#running-tests)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-yellow.svg)](LICENSE)
[![Dataset: CC BY 4.0](https://img.shields.io/badge/Dataset-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

FlowCast is an open-source deep learning forecasting system that predicts arterial highway traffic volume 15 minutes into the future across a network of 36 inductive loop detector stations. Built on historical time-series sequences and contextual calendar indicators, FlowCast pairs continuous multi-output regression with statistically derived traffic-volume bands (Low, Moderate, High) to provide interpretable traffic insights via an interactive dashboard.

---

## Table of Contents

- [What Problem FlowCast Solves](#what-problem-flowcast-solves)
- [What the Application Predicts and How the Pipeline Works](#what-the-application-predicts-and-how-the-pipeline-works)
- [Technology Stack and Responsibilities](#technology-stack-and-responsibilities)
- [Performance Benchmarks](#performance-benchmarks)
- [Prerequisites](#prerequisites)
- [Installation and Launch Instructions](#installation-and-launch-instructions)
  - [Windows (PowerShell)](#windows-powershell)
  - [macOS / Linux (Terminal)](#macos--linux-terminal)
- [First-Run Behavior and Startup Flow](#first-run-behavior-and-startup-flow)
- [Dashboard Guide](#dashboard-guide)
  - [Demonstration Mode](#demonstration-mode)
  - [Custom Scenario Builder](#custom-scenario-builder)
  - [CSV Batch Input and Schema](#csv-batch-input-and-schema)
  - [Model Comparison & Diagnostics](#model-comparison--diagnostics)
  - [Architecture & Audit](#architecture--audit)
- [Running Automated Tests](#running-automated-tests)
- [Troubleshooting](#troubleshooting)
- [Academic Documentation](#academic-documentation)
- [License and Attribution](#license-and-attribution)
- [Design Decisions: Why These Choices?](#design-decisions-why-these-choices)

---

## What Problem FlowCast Solves

Traffic congestion on metropolitan arterial corridors is highly dynamic, non-linear, and spatially interdependent. Conventional municipal operations often rely on historical static averages or reactive sensor alerts that detect congestion only after queues have formed. 

FlowCast addresses **short-horizon traffic flow forecasting** (official problem statement: *L2 — Smart Traffic Management System for Urban Congestion*). By forecasting traffic volumes 15 minutes into the future ($t+1$) from a 150-minute historical window ($t-9 \dots t$), transportation operators and downstream routing engines can anticipate congestion build-up, evaluate bottleneck propagation across adjacent sensors, and enact proactive advisory measures.

> **Academic Prototype Scope**: FlowCast is an offline machine learning benchmark developed on historical California/Northern Virginia highway loop detector data (UCI Dataset #734). It is an academic prototype, not a real-time municipal traffic signal controller or a system validated on Indian city roads.

---

## What the Application Predicts and How the Pipeline Works

1. **Prediction Target**: Simultaneously forecasts normalized traffic volume $[0.0, 1.0]$ for all **36 highway sensor stations** 15 minutes into the future ($t+1$).
2. **Volume Band Categorization**: Continuous volume predictions are translated into three operational bands (**Low Volume**, **Moderate Volume**, **High Volume**) using empirical tertile cutoffs derived strictly from the training dataset.
3. **Pipeline Stages**:
   - **Data Ingestion**: Downloads and extracts the official MATLAB archive (`traffic_dataset.mat`) from the UCI Machine Learning Repository.
   - **Feature Engineering**:
     - *Sequential branch ($X_{\text{seq}}$)*: 10 historical volume observations across all 36 sensors (shape: $10 \times 36$).
     - *Contextual branch ($X_{\text{ctx}}$)*: 29 one-hot calendar indicators (5 weekday flags + 24 hour-of-day flags).
     - *Flattened representation ($X_{\text{flat}}$)*: Exactly structured 1,728-feature vector (36 sensors $\times$ 48 features) containing historical lags, calendar indicators, and static sensor metadata (lanes, direction, road segment ID) for tree-based baselines.
   - **Model Inference**:
     - **FlowCast GRU**: Dual-branch recurrent architecture with a 2-layer GRU sequence encoder fused with a contextual MLP, mapped to 36 bounded outputs via Sigmoid activation.
     - **Random Forest Baseline**: Optimized multi-output regressor providing competitive tabular benchmark comparisons.
   - **Interactive Presentation**: Streamlit dashboard supporting demonstration instances, manual custom scenario configuration, and CSV batch uploads.

---

## Technology Stack and Responsibilities

| Technology | Responsibility & Role |
| :--- | :--- |
| **PyTorch (v2.2+)** | Core deep learning framework; powers the `FlowCastGRU` recurrent model, automatic differentiation, GPU/MPS hardware acceleration, and checkpoint serialization. |
| **Scikit-Learn** | Powers the `FlowCastBaseline` multi-output Random Forest regressor, hyperparameter grid search, and classification/regression evaluation metrics. |
| **NumPy & Pandas** | High-performance tensor manipulation, chronological sequence slicing, CSV batch dataframes, and flat vector assembly. |
| **SciPy** | MATLAB `.mat` matrix deserialization (`scipy.io.loadmat`) for loading raw sparse traffic records. |
| **Matplotlib & Seaborn** | Generation of publication-quality diagnostic plots (loss curves, sensor forecasts, confusion matrices, and model comparison bar charts). |
| **Streamlit (v1.35+)** | Interactive analytical dashboard; serves live forecasting, scenario tweaking, CSV batch testing, and diagnostic inspections. |
| **Pytest** | Automated unit and integration testing suite verifying zero-leakage splits, feature mapping, model shapes, and inference stability. |

---

## Performance Benchmarks

Evaluated on the held-out test partition ($t = 1,261 \dots 2,100$, 840 time steps, 30,240 total sensor predictions):

| Metric | Random Forest Baseline | FlowCast GRU (Dual-Branch) | Metric Interpretation |
| :--- | :---: | :---: | :--- |
| **MAE (Mean Absolute Error) ↓** | **0.0320** | **0.0371** | Average normalized volume deviation |
| **RMSE (Root Mean Squared Error) ↓** | **0.0470** | **0.0523** | Penalizes larger outlier prediction errors |
| **$R^2$ Score (Variance Explained) ↑** | **0.9471** | **0.9345** | Proportion of traffic variance captured |
| **Macro-F1 Score ↑** | **0.9057** | **0.8918** | Balanced volume band classification accuracy |
| **Overall Accuracy ↑** | **90.44%** | **89.08%** | Exact 3-band categorization concordance |

> **Tertile Volume Bands (Training-Derived)**:
> - **Low Volume Band**: Normalized volume $\le 0.1822$
> - **Moderate Volume Band**: $0.1822 < \text{Volume} \le 0.3895$
> - **High Volume Band**: Normalized volume $> 0.3895$

---

## Prerequisites

- **Operating System**: macOS, Windows 10/11, or Linux
- **Python**: Version `3.10`, `3.11`, `3.12`, or `3.13`
- **Git**: Installed and available on your system path
- **Hardware**: Runs efficiently on CPU; automatically utilizes Apple Silicon MPS or NVIDIA CUDA if present.

---

## Installation and Launch Instructions

### Windows (PowerShell)

```powershell
# 1. Clone the repository
git clone https://github.com/mdzaidibrahim/FlowCast.git
cd FlowCast

# 2. Create virtual environment
python -m venv .venv

# 3. Upgrade pip and install dependencies
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 4. Launch FlowCast (handles dataset download, training verification, and UI)
.\.venv\Scripts\python.exe run.py
```

### macOS / Linux (Terminal)

```bash
# 1. Clone the repository
git clone https://github.com/mdzaidibrahim/FlowCast.git
cd FlowCast

# 2. Create virtual environment
python3 -m venv .venv

# 3. Upgrade pip and install dependencies
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt

# 4. Launch FlowCast (handles dataset download, training verification, and UI)
./.venv/bin/python run.py
```

---

## First-Run Behavior and Startup Flow

FlowCast is designed to be fully autonomous on first execution via `run.py`:

1. **Dataset Verification**: Inspects `data/raw/traffic_dataset.mat`. If absent, it downloads the official UCI archive (~15 MB) and extracts it automatically.
2. **Artifact Verification**: Checks `artifacts/models/` and `artifacts/metrics/`. If weights are missing, it trains the Random Forest baseline and FlowCast GRU model (~10–15 seconds total).
3. **Subsequent Launches**: Reuses saved model checkpoints and demonstration fixtures immediately without retraining.
4. **Browser Launch**: Spawns the Streamlit web interface and opens `http://localhost:8501` in your default browser.

---

## Dashboard Guide

The Streamlit dashboard comprises three major screens selectable via the sidebar:

### Demonstration Mode
- **Instant Testing**: Evaluates pre-computed real instances from the held-out test partition.
- **Prediction vs Ground Truth**: Shows the forecasted volume alongside actual ground truth ($y_{\text{true}}$) and delta error ($\Delta$).
- **Focal Sensor Selector**: Focuses the primary gauge and line comparison on any of the 36 network sensors.
- **Traffic-Volume Bands**: Displays dynamic color-coded badges indicating **Low**, **Moderate**, or **High** volume risk.

### Custom Scenario Builder
- **What-If Simulations**: Allows manual adjustment of network baseline volume, focal sensor congestion spike, day of the week, and hour of the day.
- **Exact Feature Reconstruction**: Automatically populates the correct per-sensor 48-feature positions (historical lags, time encodings, lane count, road direction, and road segment ID) via `build_x_flat_from_inputs`.

### CSV Batch Input and Schema
- **Upload Custom Readings**: Upload arbitrary CSV files for instant multi-sensor forecasting.
- **Required CSV Schema**:
  - Minimum **10 rows** (representing $t-9 \dots t$ consecutive 15-minute historical steps).
  - Minimum **36 columns** (columns 0 through 35 corresponding to the 36 network sensors).
  - Values must be normalized traffic volumes between `0.0` and `1.0`.

### Model Comparison & Diagnostics
- **Side-by-Side Evaluation**: Visualizes regression metrics (MAE, RMSE, $R^2$) and classification metrics (Macro-F1, Accuracy) between FlowCast GRU and Random Forest.
- **Visual Diagnostics**: Renders training loss curves, actual vs predicted temporal series, and multi-class confusion matrices.

### Architecture & Audit
- **Pipeline Inspection**: Summarizes dataset dimensions, chronological partition boundaries, sensor station metadata, and parameter counts.
- **Zero-Leakage Verification**: Outlines strict temporal separation and threshold isolation rules.

---

## Running Automated Tests

FlowCast includes 23 automated tests covering preprocessing, zero-leakage splits, hyperparameter selection rules, custom feature flattening, CSV validation, model forward passes, and metric consistency:

```bash
# Windows
.\.venv\Scripts\python.exe -m pytest tests/ -v

# macOS / Linux
./.venv/bin/python -m pytest tests/ -v
```

---

## Troubleshooting

- **Port in Use**: If port 8501 is occupied, launch on a custom port:
  ```bash
  python run.py --port 8502
  ```
- **Force Retraining**: Retrain both models from scratch and regenerate all plots:
  ```bash
  python run.py --retrain
  ```
- **Environment Check Only**: Verify dependencies and dataset files without starting the Streamlit server:
  ```bash
  python run.py --check-only
  ```
- **SSL / Certificate Errors during Download**: The loader uses Python `certifi` and standard HTTPS fallbacks. If issues persist, download `traffic+flow+forecasting-1.zip` manually from the [UCI Repository](https://archive.ics.uci.edu/dataset/734/traffic%2Bflow%2Bforecasting) and place `traffic_dataset.mat` in `data/raw/`.

---

## Academic Documentation

| Document | Focus & Content |
| :--- | :--- |
| [`docs/literature_review.md`](docs/literature_review.md) | Verified academic literature review of 8 primary peer-reviewed works (Zhao et al., Cho et al., Chung et al., Breiman, Smith & Demetsky, Li et al., Yu et al., Wu et al.). |
| [`docs/hyperparameter_tuning.md`](docs/hyperparameter_tuning.md) | Search space, validation selection criteria, and experimental logs for 11 Random Forest trials and 8 GRU configurations. |
| [`docs/methodology.md`](docs/methodology.md) | Mathematical formulation, GRU gating dynamics, contextual MLP branch, and loss functions. |
| [`docs/dataset_audit.md`](docs/dataset_audit.md) | Comprehensive audit of UCI Dataset #734, sensor station mappings, and 48-feature column layout. |
| [`docs/report_outline.md`](docs/report_outline.md) | Academic university project report structure and chapter blueprint. |

---

## License and Attribution

- **Source Code**: Released under the open-source [MIT License](LICENSE). Copyright © 2024 Md Zaid Ibrahim.
- **Dataset Attribution**: The Traffic Flow Forecasting benchmark dataset is provided by Zhao et al. (ACM TSAS 2019 / UCI Repository ID 734) and is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
- *Note*: The code license (MIT) and the data license (CC BY 4.0) are distinct; distributing modifications of the code is governed by MIT terms.

---

## Design Decisions: Why These Choices?

| Decision | Why It Was Selected | Alternatives Considered | Trade-Offs & Limitations |
| :--- | :--- | :--- | :--- |
| **Problem Scope: 15-Minute Short-Horizon Forecasting** | Matches standard Intelligent Transportation System (ITS) control cycles and provides actionable advance notice for congestion mitigation without excessive cumulative error. | 5-minute ultra-short horizon, 60-minute long horizon, or multi-step iterative forecasting. | 15-minute aggregation dampens transient sub-minute shockwaves; 1-step lookahead does not capture multi-hour congestion dissipation. |
| **Benchmark Dataset: UCI Traffic Flow Forecasting (#734)** | Authoritative peer-reviewed spatio-temporal benchmark with 36 real highway sensors, multi-lane diversity, and real-world topological adjacency matrix. | Synthetic traffic simulators (SUMO), private local city camera feeds, or high-dimensional PeMS-04/08 grids. | Historical data from Virginia/DC highway corridors; cannot be directly transferred to mixed heterogeneous traffic in Indian cities without recalibration. |
| **Chronological Data Partitioning (Zero Leakage)** | Respects temporal causality: past ($t=0\dots1060$) trains the models, intermediate steps ($t=1061\dots1260$) select hyperparameters, and future ($t=1261\dots2100$) tests generalization. | K-fold cross-validation, random train/test splits, or rolling window cross-validation. | Random splits cause catastrophic temporal data leakage (adjacent timestamps are nearly identical); single future split tests against real distribution shifts. |
| **Baseline Model: Multi-Output Random Forest** | Robust, non-parametric ensemble baseline that handles high-dimensional colinear inputs (1,728 features) with minimal tuning and low overfitting risk. | Linear regression, ARIMA/SARIMA, Support Vector Regressors (SVR), or XGBoost. | Disregards sequential ordering within time series; requires tabular flattening; larger memory footprint when scaling to thousands of sensors. |
| **Deep Learning Architecture: Dual-Branch GRU + Contextual MLP** | Gated Recurrent Unit captures 10-step temporal inertia with ~25% fewer parameters than LSTM, avoiding vanishing gradients while contextual MLP explicitly embeds day/hour periodicity. | Standard Vanilla RNN, LSTM, Transformer, or Graph Neural Networks (DCRNN/STGCN). | Does not execute explicit graph message-passing over adjacency matrix; Random Forest baseline achieves slightly lower point error ($R^2=0.9471$ vs $0.9345$). |
| **Deep Learning Framework: PyTorch** | Dynamic computation graphs, clear Pythonic module interfaces, native Apple Silicon MPS and CUDA acceleration, and transparent checkpoint serialization. | TensorFlow/Keras, JAX, or ONNX Runtime. | Slightly more boilerplate code than Keras high-level APIs; requires custom training loop implementation. |
| **User Interface: Streamlit** | Rapid, pure-Python web deployment allowing seamless integration with PyTorch tensors, live plotting, interactive widgets, and instant model serving. | Flask/FastAPI + React frontend, Gradio, or Jupyter Notebooks. | Stateful execution reruns script on user interaction; not intended for high-concurrency production serving beyond local/internal demos. |
| **Evaluation Metrics: Dual Regression & Classification** | Evaluates both quantitative error (MAE, RMSE, $R^2$) and operational interpretability (Macro-F1, Accuracy across Low/Moderate/High bands). | Solely MAE/RMSE or raw Mean Absolute Percentage Error (MAPE). | MAPE suffers from numerical instability near near-zero volume counts; discrete tertile bands discretize continuous predictions into 3 coarse buckets. |
| **Regularization Strategy: Dropout, Weight Decay, & Early Stopping** | Prevents recurrent hidden states from memorizing the 1,061 training steps; learning rate scheduler adapts step size on plateau. | Pure L2 regularization or unregularized large models. | Requires careful validation monitoring to avoid stopping prematurely during transient loss plateaus. |
| **Automated Local Launcher (`run.py`)** | Provides zero-configuration onboarding: automatically downloads data, checks artifacts, runs tests, and launches the UI in a single command. | Manual bash setup scripts, Docker containers, or multi-step command sequences. | Requires local Python environment with correct version; Docker containerization would further isolate system dependencies. |
