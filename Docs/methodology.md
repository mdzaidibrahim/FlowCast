# FlowCast: Mathematical Methodology and Architecture Specification

## 1. Mathematical Problem Formulation
Urban short-horizon traffic forecasting operates over a spatial sensor network represented as an unweighted directed graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, where $\mathcal{V} = \{s_1, s_2, \dots, s_S\}$ denotes the set of $S = 36$ inductive loop detector stations, and $\mathcal{E}$ represents highway topological connectivity ($\mathbf{A} \in \{0, 1\}^{S \times S}$ with $|\mathcal{E}| = 86$ directed links).

At each discrete quarter-hour timestamp $t \in \{1, \dots, T\}$, the network records traffic volume readings $\mathbf{y}_t = [y_{t, 1}, y_{t, 2}, \dots, y_{t, S}]^T \in [0, 1]^S$.
Given a historical observation window of length $L = 10$ continuous 15-minute sample points:
$$\mathbf{X}_{\text{seq}, t} = [\mathbf{y}_{t-9}, \mathbf{y}_{t-8}, \dots, \mathbf{y}_t]^T \in \mathbb{R}^{L \times S}$$
along with a time-varying contextual vector $\mathbf{c}_t \in \{0, 1\}^{29}$ (composed of a 5-dimensional one-hot weekday vector $\mathbf{w}_t \in \{0, 1\}^5$ and a 24-dimensional one-hot hour-of-day vector $\mathbf{h}_t \in \{0, 1\}^{24}$):

The mathematical objective is to learn a mapping $f_\theta: (\mathbf{X}_{\text{seq}, t}, \mathbf{c}_t) \to \hat{\mathbf{y}}_{t+1} \in [0, 1]^S$ that minimizes the mean squared error over the prediction horizon $\tau = 1$ (15 minutes ahead):
$$\min_\theta \frac{1}{N \cdot S} \sum_{i=1}^N \sum_{s=1}^S \left( y_{i+1, s} - \hat{y}_{i+1, s} \right)^2 + \lambda \|\theta\|_2^2$$

---

## 2. Dual-Branch Neural Architecture

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

### 2.1 Sequential Recurrent Branch (GRU)
The Gated Recurrent Unit captures temporal traffic inertia, accumulation dynamics, and flow propagation across the 10-step sequence without vanishing gradients. For sequence step $k \in \{1, \dots, 10\}$ with input $\mathbf{x}_k \in \mathbb{R}^{36}$:

$$\mathbf{r}_k = \sigma(\mathbf{W}_r \mathbf{x}_k + \mathbf{U}_r \mathbf{h}_{k-1} + \mathbf{b}_r)$$
$$\mathbf{z}_k = \sigma(\mathbf{W}_z \mathbf{x}_k + \mathbf{U}_z \mathbf{h}_{k-1} + \mathbf{b}_z)$$
$$\tilde{\mathbf{h}}_k = \tanh(\mathbf{W}_h \mathbf{x}_k + \mathbf{U}_h (\mathbf{r}_k \odot \mathbf{h}_{k-1}) + \mathbf{b}_h)$$
$$\mathbf{h}_k = (1 - \mathbf{z}_k) \odot \mathbf{h}_{k-1} + \mathbf{z}_k \odot \tilde{\mathbf{h}}_k$$

where $\mathbf{r}_k$ is the reset gate, $\mathbf{z}_k$ is the update gate, $\tilde{\mathbf{h}}_k$ is the candidate hidden state, and $\odot$ denotes the Hadamard element-wise product.
The final hidden state $\mathbf{h}_{10} \in \mathbb{R}^{64}$ encapsulates the learned multi-sensor momentum.

### 2.2 Contextual Branch (MLP)
Temporal periodicities (morning rush hours, evening rush hours, and working day trends) govern macro-level traffic demand. The contextual branch embeds these external constraints:
$$\mathbf{h}_{\text{ctx}} = \text{ReLU}\left( \mathbf{W}_{c2} \cdot \text{Dropout}\left( \text{ReLU}(\mathbf{W}_{c1} \mathbf{c}_t + \mathbf{b}_{c1}) \right) + \mathbf{b}_{c2} \right) \in \mathbb{R}^{32}$$

### 2.3 Fusion and Bounded Activation Head
The concatenated representation $\mathbf{z}_{\text{fused}} = [\mathbf{h}_{10} \,\|\, \mathbf{h}_{\text{ctx}}] \in \mathbb{R}^{96}$ passes through the output projection head:
$$\mathbf{h}_{\text{fuse}} = \text{Dropout}\left(\text{ReLU}(\mathbf{W}_{f1} \mathbf{z}_{\text{fused}} + \mathbf{b}_{f1})\right) \in \mathbb{R}^{64}$$
$$\hat{\mathbf{y}}_{t+1} = \sigma\left(\mathbf{W}_{f2} \mathbf{h}_{\text{fuse}} + \mathbf{b}_{f2}\right) \in [0, 1]^{36}$$

The terminal `Sigmoid` activation guarantees predictions remain strictly within the physical normalized volume domain $[0.0, 1.0]$.

---

## 3. Training Protocol and Optimization
- **Loss Function**: Mean Squared Error (MSE):
  $$\mathcal{L}_{\text{MSE}} = \frac{1}{B \cdot 36} \sum_{b=1}^B \sum_{s=1}^{36} (y_{b, s} - \hat{y}_{b, s})^2$$
- **Optimizer**: Adam ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$) with initial learning rate $\eta_0 = 0.002$ and weight decay $\lambda = 10^{-4}$ ($L_2$ regularization).
- **Learning Rate Scheduling**: `ReduceLROnPlateau` monitoring validation MSE with reduction factor $\gamma = 0.5$ and patience of 4 epochs.
- **Gradient Clipping**: Norm threshold capped at $\|\mathbf{g}\|_2 \le 1.0$ to ensure recurrent numerical stability.
- **Regularization**: Dropout rates of $p = 0.15$ applied across recurrent layers and dense MLP heads.
- **Early Stopping**: Monitored on held-out validation loss with patience $P = 12$ epochs. Checkpointed weights are restored from the best validation epoch.

---

## 4. Derived Traffic-Volume Band Classification
To satisfy academic evaluation rubrics requiring discrete classification metrics without fabricating arbitrary boundaries or causing test leakage, tertile cutoffs were derived strictly from the training set target distribution:
$$\theta_{\text{low}} = \text{Percentile}_{33.33\%}(\mathbf{Y}_{\text{train}}) = 0.1822$$
$$\theta_{\text{high}} = \text{Percentile}_{66.67\%}(\mathbf{Y}_{\text{train}}) = 0.3895$$

The discrete category mapping $\mathcal{C}(y) \in \{0, 1, 2\}$ is defined as:
$$\mathcal{C}(y) = \begin{cases} 0 \quad (\text{Low Volume Band}), & \text{if } y \le 0.1822 \\ 1 \quad (\text{Moderate Volume Band}), & \text{if } 0.1822 < y \le 0.3895 \\ 2 \quad (\text{High Volume Band}), & \text{if } y > 0.3895 \end{cases}$$

This procedure applies symmetrically to ground truth labels and model predictions on the held-out test set, ensuring valid calculation of Precision, Recall, Macro-F1, and Confusion Matrices.
