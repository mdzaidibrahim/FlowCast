# FlowCast: Dataset Audit and Schema Verification Report

## 1. Executive Summary and Dataset Identification
- **Dataset Name**: Traffic Flow Forecasting
- **UCI Repository ID**: 734
- **Original Source**: UCI Machine Learning Repository / National Science Foundation
- **DOI**: [10.1145/3339823](https://doi.org/10.1145/3339823)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Academic Citation**:
  > Liang Zhao, Olga Gkountouna, and Dieter Pfoser. 2019. *Spatial Auto-regressive Dependency Interpretable Learning Based on Spatial Topological Constraints*. ACM Transactions on Spatial Algorithms and Systems (TSAS), Vol. 5, No. 3, Article 19 (August 2019), 28 pages. DOI: https://doi.org/10.1145/3339823.
- **Physical Sensor Context**: 36 loop sensor locations deployed along two major arterial highway corridors in Northern Virginia and the Washington, D.C. capital metropolitan area (Interstate 66 and Interstate 395 / Route 50 corridors).
- **Sampling Frequency**: Traffic volume recorded at continuous 15-minute intervals (quarter-hours).

---

## 2. Access Mechanism and Retrieval Verification
- **Programmatic API Status (`ucimlrepo`)**:
  Attempting `fetch_ucirepo(id=734)` raises `DatasetNotFoundError: "Traffic Flow Forecasting" dataset (id=734) exists in the repository, but is not available for import`. The UCI Python client does not currently host tabular tabular endpoints for this entry because the repository stores the dataset as a structured MATLAB archive (`traffic_dataset.mat`) inside a distribution archive.
- **Authoritative Raw Distribution**:
  Downloaded directly from UCI's public endpoint:
  `https://archive.ics.uci.edu/static/public/734/traffic+flow+forecasting-1.zip` (4.2 MB).
- **Extracted Contents**:
  - `traffic_dataset.mat` (4,401,723 bytes): Binary MATLAB v7 container storing sparse observation matrices and network graphs.
  - `Traffic Flow Prediction Dataset.docx` (16,794 bytes): Official author-provided documentation confirming matrix semantics, time intervals, spatial topology, and citations.

---

## 3. Data Dimensions, Array Types, and Original Split

| MATLAB Key | Data Structure Type | Dimensionality | Description |
| :--- | :--- | :--- | :--- |
| `tra_X_tr` | `ndarray` (object array of `scipy.sparse.csc_array`) | $(1, 1261)$ | Training input matrices for 1,261 contiguous 15-minute timestamps. Each cell contains a $(36, 48)$ sparse matrix (36 spatial locations $\times$ 48 features). |
| `tra_X_te` | `ndarray` (object array of `scipy.sparse.csc_array`) | $(1, 840)$ | Test input matrices for 840 contiguous 15-minute timestamps. Each cell contains a $(36, 48)$ sparse matrix. |
| `tra_Y_tr` | `ndarray` (dense `float64`) | $(36, 1261)$ | Training ground-truth traffic volume 15 minutes into future for all 36 sensors across 1,261 timestamps. |
| `tra_Y_te` | `ndarray` (dense `float64`) | $(36, 840)$ | Test ground-truth traffic volume 15 minutes into future for all 36 sensors across 840 timestamps. |
| `tra_adj_mat` | `ndarray` (dense `uint8`) | $(36, 36)$ | Unweighted directed spatial adjacency matrix denoting topological highway connectivity among 36 sensors (86 directed edges, 0 self-loops). |

### Sample Count and Assignment Compliance:
- **Total Continuous Quarter-Hours**: $1,261 + 840 = 2,101$ temporal steps.
- **Total Sensor-Observation Points**: $2,101 \times 36 = 75,636$ spatio-temporal readings.
- **Academic Guideline Compliance**: The assignment mandates 1,000–5,000 instances. FlowCast features exactly 2,101 temporal steps ($75,636$ total sensor observations), fully meeting and exceeding the minimum threshold without artificial data generation.

---

## 4. Temporal Ordering and Train/Test Split Boundary
Analysis of the timestamp sequence shows strict chronological continuity:
- **Training Horizon (`tra_X_tr`, `tra_Y_tr`)**: First 1,261 quarter-hours ($\approx 13.1$ days of working weekday operations).
- **Test Horizon (`tra_X_te`, `tra_Y_te`)**: Subsequent 840 quarter-hours ($\approx 8.75$ days of working weekday operations).
- **Split Ratio**: 60.02% Train ($1,261$ steps) / 39.98% Test ($840$ steps).
- **Validation Split Strategy**: To protect the author-defined held-out test set ($840$ steps) from leakage, a validation set is created by taking the final 200 time steps of the training sequence:
  - **Train split**: Steps 0 to 1,060 ($1,061$ steps, $\approx 50.5\%$ of total).
  - **Validation split**: Steps 1,061 to 1,260 ($200$ steps, $\approx 9.5\%$ of total).
  - **Held-Out Test split**: Steps 1,261 to 2,100 ($840$ steps, $\approx 40.0\%$ of total).
- **Temporal Alignment**: Verification confirms that at time step $t$, the 10 historical readings represent $[t-9, t-8, \dots, t]$, and the target $\mathbf{y}_t$ matches the sensor reading observed at $t+1$ (15 minutes ahead).

---

## 5. Feature Architecture and Semantics (48 Features per Sensor)

Each spatial observation at timestamp $t$ and sensor $s$ is represented by a 48-dimensional feature vector:

| Index Range | Width | Feature Group | Description & Encoding |
| :--- | :--- | :--- | :--- |
| `0` to `9` | 10 | **Lagged Traffic Sequence** | Traffic volume sensed during the 10 most recent 15-minute sample points ($t-9, t-8, \dots, t$). Continuous normalized values $\in [0.0, 1.0]$. |
| `10` to `14` | 5 | **Day of Week** | One-hot encoded indicators representing 5 working days (Monday, Tuesday, Wednesday, Thursday, Friday). Weekends were excluded in the primary sensing study. |
| `15` to `38` | 24 | **Hour of Day** | One-hot encoded indicators representing the 24 hours of the day (0:00 to 23:00). Exactly one bit active per timestamp. |
| `39` | 1 | **Number of Lanes** | Static physical highway attribute indicating number of travel lanes at that sensor station ($1, 2, 3, 4, 5$). |
| `40` to `43` | 4 | **Road Direction** | One-hot encoded compass direction of travel (Northbound, Southbound, Eastbound, Westbound). |
| `44` to `47` | 4 | **Road Segment ID** | One-hot encoded highway identifier representing the highway corridor section (I-66, I-395/US-50 regional segments). |

*Note on UCI Description Discrepancy*: The UCI overview webpage stated "47 features", whereas the author's accompanying documentation (`Traffic Flow Prediction Dataset.docx`) and binary matrices confirm 48 columns (10 history + 5 day-of-week + 24 hour-of-day + 1 lane count + 4 direction + 4 road ID = 48).

---

## 6. Target Analysis and Task Formulation
- **Target Variable**: Traffic volume 15 minutes into the future normalized to $[0.0, 1.0]$.
- **Target Dimensions**: At each time step $t$, the target vector is $\mathbf{y}_t \in \mathbb{R}^{36}$, containing the future traffic reading for all 36 sensors.
- **Task Formulation**: **Multi-Output Spatio-Temporal Regression**.
  - All 36 sensors are predicted simultaneously.
  - The model outputs a 36-dimensional continuous forecast for each time step.
  - For localized analysis and dashboard presentation, focal sensors (e.g., Sensor 0 along the primary I-66 corridor) can be inspected in detail alongside network-wide aggregates.
- **Statistical Distribution of Target (Training Data)**:
  - Missing Values: Exactly 0 (0.00%).
  - Minimum: $0.0000$
  - 25th Percentile: $0.1266$
  - Median (50th Percentile): $0.2933$
  - Mean: $0.3043$ ($\pm 0.1996$)
  - 75th Percentile: $0.4428$
  - 90th Percentile: $0.5764$
  - Maximum: $1.0000$

---

## 7. Non-Leaking Congestion / Traffic-Volume Category Thresholds
In accordance with academic integrity guidelines, classification categories are derived exclusively from the training set target distribution and applied uniformly to predictions and test labels:

- **Low Volume Band**: $\text{Volume} \le 0.1826$ (Training lower tertile, 33.33%).
- **Moderate Volume Band**: $0.1826 < \text{Volume} \le 0.3886$ (Training middle tertile, 33.33%).
- **High Volume Band**: $\text{Volume} > 0.3886$ (Training upper tertile, 33.34%).

*Labeling Disclaimer*: These categories represent relative traffic-volume density bands derived mathematically from the distribution percentiles of the Northern Virginia sensor network. They do not constitute statutory highway congestion thresholds or Indian Road Congress (IRC) standards.

---

## 8. Summary of Modeling Decision
1. **Baseline**: Scikit-Learn `RandomForestRegressor` trained on multi-output tabular vectors (flattened historical lags and contextual features) predicting all 36 sensor targets.
2. **Deep Learning Model**: A Dual-Branch Spatio-Temporal GRU (`FlowCastGRU`):
   - **Temporal GRU Branch**: Consumes the genuine 10-step sequence ($L=10$) of 36 sensor readings $\mathbf{X}_{\text{seq}} \in \mathbb{R}^{B \times 10 \times 36}$, preserving physical temporal causality.
   - **Context Branch**: Dense MLP embedding the 29 time context features (day + hour) and static highway features.
   - **Fusion Head**: Concatenates recurrent temporal state and contextual representations to produce multi-output forecasts $\hat{\mathbf{y}} \in \mathbb{R}^{B \times 36}$.
3. **Execution Device**: PyTorch configured to auto-detect Apple Silicon MPS (`mps`), CUDA (`cuda`), or fallback to `cpu`.
