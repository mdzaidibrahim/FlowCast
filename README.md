# FlowCast: Deep Learning-Based Short-Horizon Traffic Flow Forecasting

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

> **Problem Statement (L2)**: Smart Traffic Management System for Urban Congestion  
> **Academic Scope**: Deep Learning Mini Project (7th Semester CSE, VTU/CMRIT Curriculum)  
> **Notice**: This is an academic research prototype evaluated on highway arterial sensor datasets in Northern Virginia / Washington, D.C. It does not claim real-world signal preemption or validation on heterogeneous Indian road networks.

---

## 1. Project Overview

**FlowCast** is a reproducible, deep learning-based short-horizon traffic forecasting system. It predicts traffic volume **15 minutes into the future** across a network of **36 highway loop detector stations** using historical observations and contextual features, and translates continuous forecasts into interpretable **traffic-volume bands** (Low, Moderate, High).

Key Features:
- **Zero-Leakage Temporal Pipeline**: Enforces chronological ordering across train ($1,061$ steps), validation ($200$ steps), and held-out test ($840$ steps) sets without temporal shuffling.
- **Dual-Branch Spatio-Temporal GRU**: Consumes a genuine 10-lag historical sequence ($150$ minutes of readings) through a 2-layer GRU while encoding day/hour periodicities through an auxiliary contextual MLP.
- **Statistically Derived Categories**: Traffic-volume bands are derived strictly from training set tertiles ($0.1822$ and $0.3895$), ensuring zero test-set data leakage.
- **Interactive Streamlit Dashboard**: Provides live next-15m forecasting, sensor-level inspection, a bundled demonstration mode with real test instances, and benchmark comparisons.

---

## 2. Dataset Attribution & Audit

FlowCast utilizes the benchmark **Traffic Flow Forecasting** dataset from the UCI Machine Learning Repository:
- **UCI Dataset ID**: [734](https://archive.ics.uci.edu/dataset/734/traffic%2Bflow%2Bforecasting)
- **DOI**: [10.1145/3339823](https://doi.org/10.1145/3339823)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Citation**:
  > Liang Zhao, Olga Gkountouna, and Dieter Pfoser. 2019. *Spatial Auto-regressive Dependency Interpretable Learning Based on Spatial Topological Constraints*. ACM Transactions on Spatial Algorithms and Systems (TSAS), Vol. 5, No. 3, Article 19 (August 2019).
- **Physical Sensor Context**: 36 loop sensor locations along I-66 and I-395/Route 50 in Northern Virginia and Washington, D.C.
- **Temporal Resolution**: 15-minute sampling frequency over 2,101 continuous timestamps ($75,636$ sensor readings). Missing values: 0 (0.00%).

For full details, see [`docs/dataset_audit.md`](file:///Users/mdzaidibrahim/Dev/College%20Projects/ML%20&%20DL%20projects/DL%20mini%207th%20sem/docs/dataset_audit.md).

---

## 3. System Architecture

```
                  Input: 10-Step Historical Sequence
                    X_seq ∈ R^(B × 10 × 36)
                              │
                              ▼
                   ┌───────────────────────┐
                   │  2-Layer Gated        │
                   │  Recurrent Unit (GRU) │
                   │  Hidden Dim = 64      │
                   │  Dropout = 0.15       │
                   └───────────────────────┘
                              │
                              ▼
                   Temporal Embedding h_temp ∈ R^(B × 64)
                              │
       Input Context          │
       c_t ∈ R^(B × 29)       │
            │                 │
            ▼                 │
   ┌─────────────────┐        │
   │ Contextual MLP  │        │
   │ Linear(29, 32)  │        │
   │ ReLU + Dropout  │        │
   │ Linear(32, 32)  │        │
   │ ReLU            │        │
   └─────────────────┘        │
            │                 │
            ▼                 │
   Context Embedding h_ctx ───┴───► Spatio-Temporal Fusion
   ∈ R^(B × 32)                     Concatenation: [h_temp || h_ctx]
                                    ∈ R^(B × 96)
                                              │
                                              ▼
                                    ┌───────────────────┐
                                    │ Dense Fusion Head │
                                    │ Linear(96, 64)    │
                                    │ ReLU + Dropout    │
                                    │ Linear(64, 36)    │
                                    │ Sigmoid Output    │
                                    └───────────────────┘
                                              │
                                              ▼
                                    Forecast: y_pred ∈ [0, 1]^(B × 36)
```

---

## 4. Experimental Results

Evaluated on the held-out test horizon of **840 continuous quarter-hours** ($30,240$ individual station observations):

### Continuous Regression Metrics (Network-Wide)

| Model Architecture | MAE (↓) | RMSE (↓) | $R^2$ Score (↑) | Training Time |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest Baseline** (100 Trees, Depth 16) | **0.0320** | **0.0475** | **0.9460** | 2.02 s |
| **FlowCast GRU** (2-Layer Recurrent + Context MLP) | 0.0374 | 0.0531 | 0.9325 | 10.62 s |

### Derived Traffic-Volume Band Classification Metrics

*Thresholds derived strictly from training set tertiles: Low ($\le 0.1822$), Moderate ($0.1822 - 0.3895$), High ($> 0.3895$).*

| Model Architecture | Precision | Recall | Macro-F1 (↑) | Accuracy (↑) |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest Baseline** | **0.9065** | **0.9066** | **0.9065** | **90.53%** |
| **FlowCast GRU** | 0.8899 | 0.8893 | 0.8894 | 88.87% |

*All reported metrics are generated strictly from actual model evaluations on the held-out test set.*

---

## 5. Repository Structure

```
FlowCast/
├── app.py                      # Interactive Streamlit dashboard
├── requirements.txt            # Pinned project dependencies
├── pytest.ini                  # Pytest configuration
├── README.md                   # Project overview & documentation
├── .gitignore                  # Git exclusion rules
├── src/
│   ├── data/
│   │   ├── load_data.py        # Automated UCI dataset downloader & parser
│   │   └── preprocess.py       # Zero-leakage temporal splitter & feature extractor
│   ├── models/
│   │   ├── baseline.py         # Multi-output Random Forest regressor
│   │   ├── gru_model.py        # Spatio-Temporal Dual-Branch GRU
│   │   ├── train.py            # End-to-end training pipeline & checkpointing
│   │   └── predict.py          # Unified inference & category translation wrapper
│   └── evaluation/
│       ├── metrics.py          # Regression & classification metric computers
│       └── plots.py            # Diagnostic plotting utilities
├── tests/
│   └── test_pipeline.py        # Automated test suite (5 passing tests)
├── notebooks/
│   └── FlowCast_EDA_and_Pipeline.ipynb  # Reproducible walkthrough notebook
├── docs/
│   ├── dataset_audit.md        # Detailed schema audit and citation record
│   ├── methodology.md          # Mathematical formulation and architecture specs
│   └── report_outline.md       # 30-page university report outline
└── artifacts/
    ├── figures/                # Saved diagnostic plots
    ├── metrics/                # Raw metric evaluation summary JSON
    ├── models/                 # Serialized weights (baseline & GRU)
    └── outputs/                # Demonstration sample test instances
```

---

## 6. Getting Started & Local Execution

### 1. Clone & Set Up Virtual Environment
```bash
git clone <repo-url> FlowCast
cd FlowCast
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Automated Test Suite
```bash
pytest tests/ -v
```

### 3. Run End-to-End Training & Artifact Generation
```bash
python3 -m src.models.train
```

### 4. Launch the Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 7. Documentation & Report Outline
- [`docs/dataset_audit.md`](file:///Users/mdzaidibrahim/Dev/College%20Projects/ML%20&%20DL%20projects/DL%20mini%207th%20sem/docs/dataset_audit.md): Complete schema audit, sensor locations, feature mapping, and raw distribution analysis.
- [`docs/methodology.md`](file:///Users/mdzaidibrahim/Dev/College%20Projects/ML%20&%20DL%20projects/DL%20mini%207th%20sem/docs/methodology.md): Mathematical formulations, recurrent equations, and training dynamics.
- [`docs/report_outline.md`](file:///Users/mdzaidibrahim/Dev/College%20Projects/ML%20&%20DL%20projects/DL%20mini%207th%20sem/docs/report_outline.md): Detailed 30-page university project report outline structured for VTU/IEEE formatting.
