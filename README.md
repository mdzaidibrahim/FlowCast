# FlowCast: Deep Learning-Based Short-Horizon Traffic Flow Forecasting

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

FlowCast is an open-source deep learning forecasting system that predicts arterial highway traffic volume 15 minutes into the future across a network of 36 inductive loop detector stations. Built on historical time-series sequences and contextual calendar indicators, FlowCast pairs continuous regression with statistically derived traffic-volume bands (Low, Moderate, High) to provide interpretable traffic insights via an interactive dashboard.

---

## Key Features

- **Multi-Output Spatio-Temporal Forecasting**: Predicts next-15-minute traffic volumes simultaneously for all 36 network sensors.
- **Dual-Branch Recurrent Neural Architecture**: Consumes genuine 10-step historical sequences ($150$ minutes) via a 2-layer Gated Recurrent Unit (GRU) alongside an auxiliary Multi-Layer Perceptron (MLP) for day/hour periodicities.
- **Zero-Leakage Experimental Design**: Maintains strict chronological train/validation/test ordering and derives traffic-volume band thresholds exclusively from training data.
- **Conventional ML Benchmark**: Includes an optimized multi-output Random Forest regressor for baseline comparison.
- **Interactive Web Interface**: Streamlit dashboard with bundled demonstration scenarios, custom parameter tweaking, and CSV file upload.

---

## Technology Stack

- **Deep Learning**: PyTorch (with Apple Silicon MPS, CUDA, and CPU support)
- **Machine Learning & Metrics**: Scikit-Learn, SciPy, NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Application & Serving**: Streamlit
- **Testing**: Pytest

---

## Dataset Attribution

This project uses the public **Traffic Flow Forecasting** dataset from the UCI Machine Learning Repository:
- **Repository Link**: [UCI Dataset 734: Traffic Flow Forecasting](https://archive.ics.uci.edu/dataset/734/traffic%2Bflow%2Bforecasting)
- **DOI**: [10.1145/3339823](https://doi.org/10.1145/3339823)
- **License**: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **Citation**: Liang Zhao, Olga Gkountouna, and Dieter Pfoser. *Spatial Auto-regressive Dependency Interpretable Learning Based on Spatial Topological Constraints*. ACM Transactions on Spatial Algorithms and Systems (TSAS), Vol. 5, No. 3, Article 19 (August 2019).

*Note: The dataset represents inductive loop detectors along major highways (I-66 and I-395/US-50) in Northern Virginia and Washington, D.C. FlowCast is an academic forecasting prototype, not a production traffic-control system deployed on Indian roads.*

---

## Prerequisites

- **Git**
- **Python 3.10 – 3.13**

---

## Quick Start

### Windows (PowerShell)

```powershell
git clone https://github.com/mdzaidibrahim/FlowCast.git
cd FlowCast
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py
```

### macOS / Linux (Terminal)

```bash
git clone https://github.com/mdzaidibrahim/FlowCast.git
cd FlowCast
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python run.py
```

### What Happens on First Run

When you execute `run.py` for the first time:
1. It automatically downloads the raw UCI dataset archive from the official UCI repository and verifies file integrity.
2. If trained weights are missing, it runs the training pipeline to fit the Random Forest baseline and FlowCast GRU model (takes ~10–15 seconds on modern hardware).
3. It launches the interactive Streamlit dashboard in your default browser at `http://localhost:8501`. Subsequent runs launch the app immediately using saved model weights without retraining.

---

## Running Tests

To run the automated test suite verifying dataset loading, leakage safeguards, model shapes, and inference:

```bash
# Windows
.\.venv\Scripts\python.exe -m pytest tests/ -v

# macOS / Linux
./.venv/bin/python -m pytest tests/ -v
```

---

## Troubleshooting

- **Port in Use**: If port 8501 is already occupied, specify an alternate port:
  ```bash
  python run.py --port 8502
  ```
- **Force Retraining**: To retrain both models from scratch and regenerate evaluation plots:
  ```bash
  python run.py --retrain
  ```
- **Check Environment Without Starting UI**:
  ```bash
  python run.py --check-only
  ```
- **Windows Python Command**: If `py` is not recognized on Windows, use `python -m venv .venv` instead.

---

## Academic Documentation

- Detailed mathematical formulation and network architecture: [docs/methodology.md](docs/methodology.md)
- Complete dataset schema and sensor location audit: [docs/dataset_audit.md](docs/dataset_audit.md)
- University project report blueprint and section outline: [docs/report_outline.md](docs/report_outline.md)
