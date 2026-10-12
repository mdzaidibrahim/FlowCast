# FlowCast: University Mini Project Report Outline (30-Page Structure)

**Course**: Deep Learning Mini Project (7th Semester)  
**Problem Statement Code**: L2 — Smart Traffic Management System for Urban Congestion  
**Project Title**: FlowCast: Deep Learning-Based Short-Horizon Traffic Flow Forecasting and Congestion Risk Classification  
**Target Format**: VTU CMRIT Mini Project Guidelines / IEEE Computer Society Transactions Template  

---

## Complete Table of Contents & Page Allocation Plan (Target: 30 Pages)

| Chapter | Title and Sections | Target Page Count | Page Range |
| :--- | :--- | :---: | :---: |
| — | **Preliminary Pages** (Title, Certificate, Declaration, Acknowledgement, Abstract, Table of Contents, List of Figures, List of Tables) | 4 | pp. i – iv |
| **Chapter 1** | **Introduction and Problem Definition** (Urban congestion challenges, ITS context, Problem scope, Objectives, Project limitations) | 3 | pp. 1 – 3 |
| **Chapter 2** | **Literature Survey** (Classic time-series methods, Spatio-temporal deep learning, Graph neural networks in ITS, Summary & Research Gaps) | 4 | pp. 4 – 7 |
| **Chapter 3** | **Dataset Exploration and Preprocessing** (UCI repository attribution, Loop detector setup, EDA, Distribution analysis, Leakage-free temporal splitting) | 4 | pp. 8 – 11 |
| **Chapter 4** | **System Methodology and Architecture** (Mathematical formulation, Dual-Branch GRU architecture, Recurrent gates, Contextual MLP, Multi-output fusion head) | 5 | pp. 12 – 16 |
| **Chapter 5** | **Experimental Design and Implementation** (Baseline Random Forest, Hyperparameter optimization, Early stopping, Learning rate scheduling, Regularization) | 4 | pp. 17 – 20 |
| **Chapter 6** | **Results, Evaluation, and Comparative Analysis** (Regression metrics MAE/RMSE/R², Derived classification bands, F1-scores, Confusion matrix, Plots) | 5 | pp. 21 – 25 |
| **Chapter 7** | **Interactive Application and Deployment** (Streamlit dashboard architecture, Demonstration mode, CSV batch ingestion, Inference caching) | 2 | pp. 26 – 27 |
| **Chapter 8** | **Conclusion, Limitations, and Future Scope** (Key insights, Academic limitations, Real-world edge constraints, Future extensions) | 2 | pp. 28 – 29 |
| — | **References and Reproducibility Guide** (Peer-reviewed citations, Step-by-step reproduction instructions) | 1 | p. 30 |

---

## Detailed Section Breakdown and Technical Content

### Preliminary Pages (pp. i – iv)
- **Title Page**: College header, Department of Computer Science & Engineering, project title, guide name, student names and USN.
- **Certificate & Declaration**: Official institutional certificate certifying bonafide work.
- **Abstract**:
  > Urban vehicular congestion leads to severe economic losses, heightened carbon emissions, and transit unpredictability. Accurate short-horizon traffic forecasting enables proactive traffic demand mitigation and dynamic navigational guidance. In this work, we present **FlowCast**, an academic prototype for multi-output short-horizon traffic flow forecasting and congestion risk categorization. Utilizing the benchmark UCI Traffic Flow Forecasting dataset collected across 36 inductive loop sensor stations along Northern Virginia arterial corridors, FlowCast implements a Dual-Branch Spatio-Temporal Gated Recurrent Unit (GRU) neural network. The recurrent branch consumes a genuine 10-lag historical sequence (150 minutes of 15-minute observations) across all 36 stations, while an auxiliary Multi-Layer Perceptron (MLP) encodes 29 contextual features encompassing day-of-week and hour-of-day dynamics. Over a held-out test horizon of 840 contiguous quarter-hours (30,240 station observations), FlowCast achieves a Mean Absolute Error (MAE) of **0.0374**, a Root Mean Squared Error (RMSE) of **0.0531**, and an **$R^2$ score of 0.9325**, competitive with an optimized Random Forest benchmark (MAE **0.0320**, $R^2$ **0.9460**). To bridge continuous forecasting with operational decision-making, we derive non-leaking traffic-volume bands (Low, Moderate, High) derived strictly from training set tertiles ($0.1822$ and $0.3895$). The GRU achieves an overall classification accuracy of **88.87%** and a **Macro-F1 score of 0.8894** (Random Forest: **90.53%** and **0.9065**). A lightweight, interactive Streamlit dashboard is developed to provide live next-15-minute volume forecasting, sensor-level exploration, and bundled scenario demonstration. We explicitly emphasize that FlowCast is an academic forecasting prototype evaluated on U.S. highway corridor sensors, not a deployed traffic signal control system validated on heterogeneous Indian road networks.

---

### Chapter 1: Introduction and Problem Definition (pp. 1 – 3)
1. **1.1 Background and Motivation**: Urbanization and vehicular density increase; costs of congestion; role of Intelligent Transportation Systems (ITS).
2. **1.2 Problem Definition (L2)**: Short-horizon volume forecasting ($t+1$ where $\Delta t = 15\text{ min}$) using preceding observations ($t-9 \dots t$) and contextual indicators.
3. **1.3 Project Scope & Objectives**:
   - Multi-output spatio-temporal forecasting across a sensor network.
   - Non-leaking discrete traffic band categorization.
   - Benchmarking conventional ML against deep recurrent modeling.
   - Web interface for interactive academic evaluation.
4. **1.4 Explicit Disclosures and Limitations**: The model reflects fixed-lane highway loops in Northern Virginia and Washington, D.C.; does not integrate heterogeneous non-lane based traffic, pedestrian crossings, or live traffic signal actuators.

---

### Chapter 2: Literature Survey (pp. 4 – 7)
1. **2.1 Classical Time-Series Models**:
   - Autoregressive Integrated Moving Average (ARIMA) and seasonal extensions (Williams & Hoel, 2003). Limitations: linear stationarity assumptions, high computational scaling for multi-node networks.
   - Support Vector Regression (SVR) and Random Forests (Smith et al., 2002; Liaw & Wiener, 2002). Strengths in tabular feature non-linearities; limitation in preserving physical temporal sequentiality.
2. **2.2 Recurrent Neural Networks in Traffic Flow**:
   - Long Short-Term Memory (LSTM) (Hochreiter & Schmidhuber, 1997; Ma et al., 2015).
   - Gated Recurrent Units (GRU) (Cho et al., 2014; Chung et al., 2014). Advantages of GRU: parameter efficiency, faster convergence, and robust vanishing-gradient mitigation on shorter sequences ($L=10$).
3. **2.3 Spatio-Temporal and Graph Neural Networks**:
   - DCRNN (Diffusion Convolutional Recurrent Neural Network, Li et al., 2018).
   - STGCN (Spatio-Temporal Graph Convolutional Networks, Yu et al., 2018).
   - Spatial Autoregressive Topological Learning (Zhao et al., 2019 - primary dataset paper).
4. **2.4 Summary of Research Gaps & FlowCast Positioning**: Avoiding common tutorial antipatterns (such as reshaping unrelated static features into false temporal sequences); enforcing mathematical boundary constraints ($[0, 1]$ output domain); and deriving statistically defensible classification bands.

---

### Chapter 3: Dataset Exploration and Preprocessing (pp. 8 – 11)
1. **3.1 Dataset Attribution & Provenance**:
   - UCI Machine Learning Repository ID: 734.
   - Authors: Liang Zhao, Olga Gkountouna, and Dieter Pfoser (ACM TSAS, 2019).
   - Sensor setup: 36 loop detector stations along I-66 and I-395/Route 50 corridors.
2. **3.2 Schema and Feature Verification**:
   - 48 features: 10 historical traffic readings ($t-9 \dots t$), 5 weekday flags, 24 hour flags, 1 lane count, 4 direction indicators, 4 road section indicators.
   - Ground truth target $\mathbf{y}_{t+1} \in \mathbb{R}^{36}$.
   - Spatial highway adjacency matrix $\mathbf{A} \in \{0, 1\}^{36 \times 36}$ (86 directed edges).
3. **3.3 Exploratory Data Analysis (EDA)**:
   - Target statistics: Min $= 0.0000$, Max $= 1.0000$, Mean $= 0.3043$, Median $= 0.2933$, Std $= 0.1996$. Missing values: $0$ ($0.00\%$).
   - Diurnal traffic curves: Morning peak (07:00–09:00) and evening peak (16:30–18:30).
4. **3.4 Zero-Leakage Chronological Data Partitioning**:
   - Preserved author temporal ordering without shuffling:
     - Training Split: $t = 0 \dots 1,060$ ($1,061$ quarter-hours, $\approx 50.5\%$).
     - Validation Split: $t = 1,061 \dots 1,260$ ($200$ quarter-hours, $\approx 9.5\%$).
     - Held-Out Test Split: $t = 1,261 \dots 2,100$ ($840$ quarter-hours, $\approx 40.0\%$).

---

### Chapter 4: System Methodology and Architecture (pp. 12 – 16)
1. **4.1 Spatio-Temporal Problem Formulation**:
   - Input sequence tensor $\mathbf{X}_{\text{seq}} \in \mathbb{R}^{B \times 10 \times 36}$.
   - Context vector $\mathbf{X}_{\text{ctx}} \in \mathbb{R}^{B \times 29}$.
   - Target vector $\mathbf{Y} \in [0, 1]^{B \times 36}$.
2. **4.2 FlowCast Dual-Branch Neural Architecture**:
   - **Branch 1 (Recurrent Temporal Branch)**: 2-layer GRU (input dim $= 36$, hidden dim $= 64$, dropout $= 0.15$). Operates along sequence dimension $L=10$.
   - **Branch 2 (Contextual Auxiliary Branch)**: 2-layer MLP (input dim $= 29$, hidden dim $= 32$, ReLU activation, dropout $= 0.15$).
   - **Branch 3 (Fusion & Output Head)**: Concatenation ($64 + 32 = 96$) $\to$ Linear(96, 64) $\to$ ReLU $\to$ Dropout(0.15) $\to$ Linear(64, 36) $\to$ Sigmoid.
3. **4.3 Mathematical Justification for Sigmoid Activation**: Enforces output bounds $\hat{y} \in (0, 1)$, preventing physically impossible negative traffic volumes without artificial heuristic post-processing.
4. **4.4 Adjacency Matrix Integration**: Analysis of the 86 topological edges connecting the 36 sensor stations.

---

### Chapter 5: Experimental Design and Implementation (pp. 17 – 20)
1. **5.1 Conventional Baseline (Random Forest Regressor)**:
   - 100 decision trees, max depth $= 16$, random seed $= 42$. Multi-output regression across all 36 stations on 1,728 tabular inputs.
2. **5.2 Training Infrastructure & Hardware Acceleration**:
   - Implementation in PyTorch 2.14.
   - Hardware detection: Apple Silicon Metal Performance Shaders (`mps`) accelerator (training completed in 10.6 seconds).
3. **5.3 Regularization & Training Dynamics**:
   - Adam optimizer ($\eta = 0.002$, weight decay $= 10^{-4}$).
   - `ReduceLROnPlateau` scheduler (patience $= 4$, decay factor $= 0.5$).
   - Gradient norm clipping ($\le 1.0$).
   - Early stopping (patience $= 12$) terminating at epoch 35, restoring best checkpoint from epoch 23 (Val MSE: $0.00170$).

---

### Chapter 6: Results, Evaluation, and Comparative Analysis (pp. 21 – 25)
1. **6.1 Continuous Regression Performance**:
   - Comparative Table on Held-Out Test Set ($N = 840$ steps, $30,240$ observations):
     - **Random Forest**: MAE $= 0.0320$, RMSE $= 0.0475$, $R^2 = 0.9460$, Training time $= 2.02\text{ s}$.
     - **FlowCast GRU**: MAE $= 0.0374$, RMSE $= 0.0531$, $R^2 = 0.9325$, Training time $= 10.62\text{ s}$.
   - Analysis: Both models exhibit strong predictive capacity ($R^2 > 0.93$), with Random Forest marginally leading in point regression, while GRU captures temporal momentum with fewer parameters ($85\text{k}$ parameters vs tens of thousands of tree splits).
2. **6.2 Derived Traffic-Volume Band Classification**:
   - Thresholds strictly from training tertiles: Low ($\le 0.1822$), Moderate ($0.1822 - 0.3895$), High ($> 0.3895$).
   - **Random Forest**: Precision $= 0.9065$, Recall $= 0.9066$, Macro-F1 $= 0.9065$, Accuracy $= 90.53\%$.
   - **FlowCast GRU**: Precision $= 0.8899$, Recall $= 0.8893$, Macro-F1 $= 0.8894$, Accuracy $= 88.87\%$.
3. **6.3 Visualizations & Diagnostic Analysis**:
   - Training vs Validation loss curve (Figure 6.1): Smooth monotonic decrease and clean plateau without divergence.
   - Actual vs Predicted volume for Sensor 0 (Figure 6.2): Scatter fidelity along $y=x$ and time-series tracking.
   - Confusion matrices (Figures 6.3 & 6.4): High diagonal concentration across Low, Moderate, and High traffic bands.

---

### Chapter 7: Interactive Application and Deployment (pp. 26 – 27)
1. **7.1 Streamlit Dashboard Architecture**:
   - Model caching (`@st.cache_resource`) preventing repeated disk loading or on-the-fly retraining.
   - Demonstration Mode: Instantly loads real test instances across peak morning, midday, evening, and night hours.
   - Custom Scenario Mode: Sliders for temporal indicators and focal sensor recent history.
   - CSV Batch Upload: Automatic schema validation and prediction.
2. **7.2 Multi-Sensor Network Visualization**:
   - 36-station forecast bar chart, focal sensor trend curve, and traffic band badge.

---

### Chapter 8: Conclusion, Limitations, and Future Scope (pp. 28 – 29)
1. **8.1 Key Findings & Conclusion**:
   - Successfully proved feasible short-horizon multi-output forecasting using dual-branch GRU.
   - Zero-leakage temporal design ensures realistic academic metrics.
2. **8.2 Academic & Operational Limitations**:
   - Dataset represents fixed inductive loops on U.S. freeways.
   - Does not model weather incidents, lane closures, or sudden accidents.
3. **8.3 Future Scope**:
   - Graph Attention Networks (GAT) to dynamically weight spatial adjacency matrix $\mathbf{A}$.
   - Multi-step horizon forecasting ($30\text{ min}$, $45\text{ min}$, $60\text{ min}$).
   - Adaptation to Indian urban arterial datasets (e.g. video-based vehicle detection).

---

### References (p. 30)
1. Liang Zhao, Olga Gkountouna, and Dieter Pfoser. 2019. Spatial Auto-regressive Dependency Interpretable Learning Based on Spatial Topological Constraints. *ACM Trans. Spatial Algorithms Syst.* 5, 3, Article 19 (August 2019).
2. K. Cho, B. van Merriënboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio. 2014. Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. *EMNLP 2014*.
3. J. Chung, C. Gulcehre, K. Cho, and Y. Bengio. 2014. Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling. *NIPS 2014 Workshop*.
4. Y. Li, R. Yu, C. Shahabi, and Y. Liu. 2018. Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting. *ICLR 2018*.
5. B. M. Smith, B. T. Demetsky. 2002. Short-term traffic flow prediction: considering utility. *IEEE Trans. Intell. Transp. Syst.* 3(4): 253-264.
6. B. M. Williams, L. A. Hoel. 2003. Modeling and forecasting vehicular traffic flow as a seasonal ARIMA process. *J. Transp. Eng.* 129(6): 664-672.
