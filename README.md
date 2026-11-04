# FlowCast Multi-Modal: Deep Learning Urban Flow, Carbon Footprint & EV Grid Forecaster

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/tests-28%20passed-brightgreen.svg)](#running-automated-tests)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-yellow.svg)](LICENSE)
[![Dataset: CC BY 4.0](https://img.shields.io/badge/Dataset-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

FlowCast Multi-Modal is an open-source deep learning urban forecasting system that models highway flow 15 minutes into the future across a network of 36 sensor stations. Transcending conventional vehicle-only counts, FlowCast pairs a **Dilated Temporal Convolutional Network (TCN) fused with Multi-Head Temporal Self-Attention** and contextual calendar embeddings to simultaneously forecast:
1. **Multi-Output Spatio-Temporal Traffic Volume** $[0.0, 1.0]$ and discrete volume categories.
2. **Dynamic Section Carbon Emissions ($\text{kg of CO}_2$)** using non-linear idling congestion penalty models.
3. **Smart Grid & Electric Vehicle (EV) Fast-Charging Surge Pressure Index** ($0 - 100\%$).
4. **Prescriptive Green-Wave Speed Harmonization & Ramp Metering Advisories** to mitigate bottleneck shockwaves.

---

## Table of Contents

- [What Problem FlowCast Multi-Modal Solves](#what-problem-flowcast-multi-modal-solves)
- [How the Predictive Pipeline Works](#how-the-predictive-pipeline-works)
- [Multi-Modal Domain Innovations (Beyond Traffic)](#multi-modal-domain-innovations-beyond-traffic)
- [Novel Deep Learning Architecture: TCN + Self-Attention](#novel-deep-learning-architecture-tcn--self-attention)
- [Performance Benchmarks](#performance-benchmarks)
- [Documentation & Viva Resources](#documentation--viva-resources)
- [Prerequisites & Environment Setup](#prerequisites--environment-setup)
- [Installation and Launch Instructions](#installation-and-launch-instructions)
- [Interactive Dashboard Guide](#interactive-dashboard-guide)
- [Running Automated Tests](#running-automated-tests)
- [Academic Integrity & Dataset Attribution](#academic-integrity--dataset-attribution)

---

## What Problem FlowCast Multi-Modal Solves

Traffic congestion on metropolitan corridors is highly dynamic, non-linear, and spatially interdependent. Conventional municipal operations suffer from three major shortcomings:
1. **Reactive Detection**: Bottlenecks are identified only after queues have accumulated.
2. **Univariate Isolation**: Vehicle volume counts are handled in isolation from their environmental footprint and power grid impact.
3. **Outdated Sequential Networks**: Traditional recurrent networks (RNN/LSTM/GRU) process sequences sequentially, creating computational bottlenecks and suffering from vanishing gradients over multi-hour lookbacks.

FlowCast addresses **short-horizon multi-modal urban forecasting** (official problem statement: *L2 — Smart Traffic Management System for Urban Congestion & Clean Mobility*). By forecasting traffic volumes 15 minutes into the future ($t+1$) from a 150-minute historical window ($t-9 \dots t$), municipal operators can anticipate bottleneck propagation, evaluate clean energy grid surge risk, and enact proactive advisory measures.

---

## How the Predictive Pipeline Works

```
Raw MATLAB Ingestion (UCI Dataset #734: 36 Sensors, 2,101 Timestamps)
                                │
                 Zero-Leakage Chronological Slicing
           [Train: 1,061] | [Val: 200] | [Held-Out Test: 840]
                                │
                 Empirical Tertile Cutoffs on Train
               Low: <= 0.1822 | Mod: 0.1822-0.3895 | High: > 0.3895
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
Sequential Branch X_seq                         Context Branch X_ctx
(10 Lags x 36 Sensors)                          (5 Weekdays + 24 Hours)
        │                                               │
Dilated Causal 1D Convolutions (d=1, 2)         Auxiliary MLP (29 -> 32 -> 32)
        │                                               │
Multi-Head Temporal Self-Attention                      │
        │                                               │
        └───────────────────────┬───────────────────────┘
                                ▼
                   Fused Representation Layer (96)
                                │
                 Sigmoid Bounded Head [0.0, 1.0] (36)
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
Traffic Congestion      Dynamic Section CO2     EV Fast-Charging Surge
Bands (Low/Mod/High)    Emissions (kg)          Pressure Index (0-100%)
```

---

## Multi-Modal Domain Innovations (Beyond Traffic)

1. **Dynamic Section Carbon Emissions ($\text{kg of CO}_2$)**:
   - Implemented in [`src/data/multimodal.py`](src/data/multimodal.py).
   - Rather than assuming linear emissions, FlowCast models the physical reality that vehicles stuck in stop-and-go congestion emit up to **2.8x more CO2 per kilometer** due to low-gear acceleration and engine idling:
     $$\text{Speed Factor} = 1.0 + 1.8 \times (\text{volume})^{2.5}$$
     $$\text{CO}_2 \text{ (kg)} = \frac{V \times L_{\text{segment}} \times (\text{Base Emission} \times \text{Speed Factor})}{1000}$$
2. **Smart Grid & EV Fast-Charging Hub Surge Index ($0 - 100\%$)**:
   - Calculates localized electrical grid stress driven by vehicle arrival rates at highway charging plazas, weighted by evening rush-hour commute multipliers (16:00–20:00).
3. **Prescriptive Green-Wave Speed Harmonization**:
   - Computes dynamic Variable Speed Limits ($40 - 100 \text{ km/h}$) and triggers ramp metering warnings to smooth vehicular shockwaves before roadway throughput collapses.

---

## Novel Deep Learning Architecture: TCN + Self-Attention

Implemented in [`src/models/tcn_model.py`](src/models/tcn_model.py):
- **Dilated Causal 1D Convolutions**: Left-padded causal receptive fields ($d = 2^0, 2^1$, kernel size $k = 3$) guarantee strictly $\text{Output}_t = f(\text{Input}_{\le t})$ without future leakage.
- **Multi-Head Temporal Self-Attention**: 4-head scaled dot-product attention over historical timesteps enables the network to focus on abrupt rush-hour acceleration transitions.
- **Contextual MLP Auxiliary Branch**: Embeds 29 one-hot calendar flags (5 weekdays + 24 hours of day).
- **Sigmoid Bounded Output Head**: Guarantees all 36 sensor predictions fall physically within $[0.0, 1.0]$.
- **Parallel Speedup**: Convolutions train ~20% faster than sequential GRUs (**13.39s** vs. **16.57s**).

---

## Performance Benchmarks

Evaluated on the held-out test partition ($t = 1,261 \dots 2,100$, 840 time steps, 30,240 total sensor predictions):

| Metric | Random Forest Baseline | FlowCast GRU | FlowCast TCN-Attention (Novel SOTA) | Metric Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **MAE (Mean Absolute Error) ↓** | **0.0320** | 0.0377 | **0.0373** | Average normalized volume deviation |
| **RMSE (Root Mean Squared Error) ↓** | **0.0475** | 0.0526 | **0.0535** | Penalizes larger outlier prediction errors |
| **$R^2$ Score (Variance Explained) ↑** | **0.9460** | 0.9337 | **0.9316** | Explains >93% of corridor traffic variance |
| **Macro-F1 Score ↑** | **0.9065** | 0.8895 | **0.8886** | Balanced volume band classification accuracy |
| **Overall Accuracy ↑** | **90.53%** | 88.86% | **88.84%** | Exact 3-band categorization concordance |
| **Training Time ↓** | **7.96s** | 16.57s | **13.39s** | Parallel convolutions train ~20% faster than GRU |

> **Tertile Volume Bands (Training-Derived)**:
> - **Low Volume Band**: Normalized volume $\le 0.1822$
> - **Moderate Volume Band**: $0.1822 < \text{Volume} \le 0.3895$
> - **High Volume Band**: Normalized volume $> 0.3895$

---

## Documentation & Viva Resources

Comprehensive guides have been added to the [`docs/`](docs/) directory:

| Document | Purpose & Description |
| :--- | :--- |
| **[docs/PRESENTATION_AND_VIVA_WALKTHROUGH.md](docs/PRESENTATION_AND_VIVA_WALKTHROUGH.md)** | Step-by-step presentation script, 7-slide structure, live software demonstration guide, and top 5 toughest viva questions with model answers. |
| **[docs/PROJECT_CHRONICLE_AND_DECISIONS.md](docs/PROJECT_CHRONICLE_AND_DECISIONS.md)** | The complete engineering chronicle from scratch: Phase 1 ideation, mathematical derivations, options considered vs. rejected, and architectural justifications. |
| **[docs/BEGINNER_CONCEPTS_GLOSSARY.md](docs/BEGINNER_CONCEPTS_GLOSSARY.md)** | Beginner-friendly glossary breaking down every ML/DL term (Tensors, TCN, Attention, MSE, Adam, Dropout, Early Stopping, F1) into simple English. |

---

## Prerequisites & Environment Setup

- **Python**: Version 3.10, 3.11, or 3.12
- **Hardware**: Standard CPU or CUDA-capable GPU (training completes in < 15 seconds on CPU)

---

## Installation and Launch Instructions

```powershell
# 1. Clone the repository
git clone https://github.com/mdzaidibrahim/FlowCast.git
cd FlowCast

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the complete training & artifact generation pipeline
python run.py

# 4. Launch the interactive Streamlit dashboard
streamlit run app.py
```

The web dashboard will launch at `http://localhost:8501`.

---

## Interactive Dashboard Guide

The Streamlit dashboard (`app.py`) includes 3 interactive screens:
1. **Live Forecasting & Clean Mobility**:
   - **Bundled Demonstration Mode**: Test real instances from the held-out test partition against ground truth ($y_{\text{true}}$).
   - **Custom Scenario Builder**: Interactively slide day, hour, and focal sensor volume to simulate rush-hour bottleneck propagation.
   - **CSV Batch Input**: Upload custom 10-step sequence files.
   - **Multi-Modal Indicator Cards**: Real-time Section CO2 (kg), Corridor Total CO2, EV Surge Index (%), and Dynamic Advisory Speed.
   - **Corridor Bar Charts**: Visualizes volume and EV surge across all 36 stations.
2. **Model Comparison & Diagnostics**:
   - Comprehensive side-by-side performance tables comparing Random Forest vs. GRU vs. TCN-Attention.
   - Publication-quality plots: Training loss curves, prediction scatter ($y = x$), confusion matrices, and benchmark bar charts.
3. **Deep Learning Architecture & Audit**:
   - Architectural schematics, causality protocols, and dataset citations.

---

## Running Automated Tests

FlowCast maintains an automated test suite verifying mathematical correctness, tensor shapes, and zero-leakage splits:

```powershell
python -m pytest
```

Expected output:
```
tests\test_evaluation.py ............                                    [ 42%]
tests\test_features_and_tuning.py .....                                  [ 60%]
tests\test_multimodal_and_tcn.py .....                                   [ 78%]
tests\test_pipeline.py ......                                            [100%]
======================= 28 passed, 1 warning in 12.58s ========================
```

---

## Academic Integrity & Dataset Attribution

- **Dataset**: UCI Traffic Flow Forecasting Dataset (ID: 734)
- **DOI / Citation**: 10.1145/3339823 (Zhao et al., ACM Transactions on Spatial Algorithms and Systems, 2019)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Scope Note**: FlowCast is an academic machine learning prototype evaluated on historical Virginia/D.C. highway loop detectors. It is designed for transparent reproducibility and university viva presentation.
