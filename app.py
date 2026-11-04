"""FlowCast Multi-Modal: Deep Learning-Based Urban Flow, Carbon Footprint & EV Grid Forecaster.

Streamlit interactive dashboard featuring:
- Dilated Temporal Convolutional Network (TCN) + Multi-Head Temporal Self-Attention
- Dual-Branch GRU & Multi-Output Random Forest Baselines
- Multi-Modal Urban Features: Real-time CO2 emissions (kg) & Smart Grid EV Fast-Charging Surge Pressure
- Prescriptive Green-Wave Speed Harmonization & Ramp Metering Advisories
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.data.multimodal import (
    compute_carbon_emissions,
    compute_ev_grid_surge_index,
    compute_green_wave_advisory,
)
from src.data.preprocess import build_x_flat_from_inputs, classify_traffic
from src.models.predict import FlowCastPredictor

# Page configuration
st.set_page_config(
    page_title="FlowCast Multi-Modal | Deep Learning Urban Flow & Clean Mobility",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished scientific appearance
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0d47a1;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #455a64;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 14px;
        border-left: 4px solid #0d47a1;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .band-low {
        color: #1b5e20;
        font-weight: bold;
        background-color: #e8f5e9;
        padding: 4px 8px;
        border-radius: 4px;
    }
    .band-mod {
        color: #e65100;
        font-weight: bold;
        background-color: #fff3e0;
        padding: 4px 8px;
        border-radius: 4px;
    }
    .band-high {
        color: #b71c1c;
        font-weight: bold;
        background-color: #ffebee;
        padding: 4px 8px;
        border-radius: 4px;
    }
    .carbon-tag {
        color: #37474f;
        background-color: #eceff1;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .surge-tag {
        color: #4a148c;
        background-color: #f3e5f5;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictor():
    """Load cached inference engine without retraining."""
    return FlowCastPredictor(
        tcn_path="artifacts/models/tcn_attention_best.pth",
        gru_path="artifacts/models/gru_best.pth",
        rf_path="artifacts/models/baseline_rf.joblib",
        metrics_path="artifacts/metrics/metrics_summary.json",
    )


@st.cache_data
def load_metrics_and_samples():
    """Load cached experiment metrics, sample demonstration inputs, and sensor metadata."""
    metrics_file = Path("artifacts/metrics/metrics_summary.json")
    samples_file = Path("artifacts/outputs/demo_samples.json")
    meta_file = Path("artifacts/outputs/sensor_metadata.json")

    metrics_data = {}
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

    samples_data = []
    if samples_file.exists():
        with open(samples_file, "r", encoding="utf-8") as f:
            samples_data = json.load(f)

    sensor_metadata = {}
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            sensor_metadata = json.load(f)
    else:
        try:
            from src.data.load_data import load_raw_traffic_data
            raw = load_raw_traffic_data()
            X0 = raw["X_tr_raw"][0]
            sensor_metadata = {
                "num_lanes": X0[:, 39].astype(int).tolist(),
                "road_direction": np.argmax(X0[:, 40:44], axis=1).tolist(),
                "road_id": np.argmax(X0[:, 44:48], axis=1).tolist(),
            }
        except Exception:
            sensor_metadata = {
                "num_lanes": [3] * 36,
                "road_direction": [0] * 36,
                "road_id": [0] * 36,
            }

    return metrics_data, samples_data, sensor_metadata


def main():
    predictor = load_predictor()
    metrics_data, samples_data, sensor_metadata = load_metrics_and_samples()

    st.sidebar.image(
        "https://images.unsplash.com/photo-1506521781263-d8422e82f27a?auto=format&fit=crop&w=400&q=80",
        caption="FlowCast Multi-Modal Urban Intelligence",
        use_container_width=True,
    )
    st.sidebar.title("Navigation & Parameters")
    nav_option = st.sidebar.radio(
        "Select Screen",
        [
            "Live Forecasting & Clean Mobility",
            "Model Comparison & Diagnostics",
            "Deep Learning Architecture & Audit",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Forecasting Engine Settings")
    active_model = st.sidebar.selectbox(
        "Active Deep Learning Architecture",
        [
            "FlowCast TCN-Attention (Novel SOTA)",
            "FlowCast GRU (Recurrent Baseline)",
            "Random Forest Baseline",
        ],
    )
    focal_sensor = st.sidebar.slider("Focal Sensor Station", min_value=0, max_value=35, value=0, step=1)

    low_th, high_th = predictor.thresholds
    st.sidebar.caption(
        rf"**Classification Thresholds (Train Derived):**" "\n"
        rf"- Low Band: $\le {low_th:.4f}$" "\n"
        rf"- Moderate Band: $({low_th:.4f}, {high_th:.4f}]$" "\n"
        rf"- High Band: $> {high_th:.4f}$"
    )

    if nav_option == "Live Forecasting & Clean Mobility":
        render_forecasting_dashboard(predictor, samples_data, sensor_metadata, active_model, focal_sensor)
    elif nav_option == "Model Comparison & Diagnostics":
        render_model_comparison(metrics_data)
    else:
        render_architecture_and_audit(metrics_data)


def render_forecasting_dashboard(predictor, samples_data, sensor_metadata, active_model, focal_sensor):
    st.markdown('<div class="main-title">FlowCast Multi-Modal: Traffic, CO₂ & Smart Grid Forecaster</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">15-Minute Spatio-Temporal Prediction with Dilated TCN, Self-Attention, Carbon Footprint & EV Grid Surge Modeling</div>',
        unsafe_allow_html=True,
    )

    with st.expander("ℹ️ Academic Scope & Novel Multi-Modal Features", expanded=False):
        st.markdown(
            """
            **What Makes This Model Unique Beyond Ordinary Traffic Forecasting:**
            - **Dilated TCN + Temporal Multi-Head Attention:** Replaces standard RNNs with dilated causal convolutions and multi-head attention to model long-range temporal dependencies without vanishing gradients.
            - **Dynamic CO₂ Emission Impact:** Quantifies real-time kilograms of carbon dioxide emitted across corridor links, accounting for vehicle idling non-linearities during bottleneck saturation.
            - **Smart Grid & EV Fast-Charging Surge Index:** Calculates real-time electrical grid strain (0–100%) driven by arrival rates at corridor fast-charging hubs.
            - **Prescriptive Green-Wave Speed Harmonization:** Recommends variable advisory speeds (40–100 km/h) and triggers ramp metering alerts to extinguish shockwaves before queues form.
            """
        )

    tab_demo, tab_custom, tab_upload = st.tabs([
        "🚀 Bundled Demonstration Mode",
        "⚙️ Custom Scenario Builder",
        "📂 CSV Batch Input",
    ])

    x_seq, x_ctx, x_flat, y_actual = None, None, None, None
    sel_hour = 17

    with tab_demo:
        st.write("Evaluate live inference on real held-out test steps from the benchmark dataset:")
        if samples_data:
            demo_names = [
                f"Test Step t={s['test_sample_index']} (Actual Corridor Vol: {s['actual_mean_volume']:.3f})"
                for s in samples_data
            ]
            selected_idx = st.selectbox("Select Benchmark Scenario:", range(len(samples_data)), format_func=lambda i: demo_names[i])
            curr_s = samples_data[selected_idx]
            y_actual = np.array(curr_s["sensor_actual"])

            # Load matching raw sequence from test split
            from src.data.preprocess import prepare_data_splits
            splits = prepare_data_splits(val_size=200)
            idx = curr_s["test_sample_index"]
            x_seq = splits.X_test_seq[idx]
            x_ctx = splits.X_test_ctx[idx]
            x_flat = splits.X_test_flat[idx]
            # Extract hour
            sel_hour = int(np.argmax(x_ctx[5:29]))
        else:
            st.warning("Demo samples not loaded. Using default synthetic sample.")
            x_seq = np.full((10, 36), 0.35, dtype=np.float32)
            x_ctx = np.zeros(29, dtype=np.float32)
            x_ctx[0] = 1.0
            x_ctx[5 + 17] = 1.0
            x_flat = build_x_flat_from_inputs(x_seq, x_ctx, sensor_metadata)

    with tab_custom:
        st.write("Configure custom temporal and spatial parameters to simulate traffic spikes or off-peak hours:")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            sel_day = st.selectbox("Day of Week:", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
            sel_hour = st.slider("Hour of Day (0-23):", 0, 23, 17)
        with col_c2:
            base_vol = st.slider("Network Base Traffic Volume:", 0.0, 1.0, 0.40, 0.05)
            focal_vol = st.slider(f"Focal Sensor #{focal_sensor} Peak Volume:", 0.0, 1.0, 0.65, 0.05)

        if st.button("Simulate Custom Scenario"):
            x_seq = np.full((10, 36), base_vol, dtype=np.float32)
            x_seq[:, focal_sensor] = np.linspace(focal_vol * 0.75, focal_vol, 10)
            x_ctx = np.zeros(29, dtype=np.float32)
            day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4}
            x_ctx[day_map[sel_day]] = 1.0
            x_ctx[5 + sel_hour] = 1.0
            x_flat = build_x_flat_from_inputs(x_seq, x_ctx, sensor_metadata)

    with tab_upload:
        st.write("Upload custom CSV containing 10 historical timesteps across all 36 sensor stations:")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if df.shape[1] >= 36 and len(df) >= 10:
                    x_seq = df.iloc[:10, :36].values.astype(np.float32)
                    x_ctx = np.zeros(29, dtype=np.float32)
                    x_ctx[0] = 1.0
                    x_ctx[5 + 17] = 1.0
                    x_flat = build_x_flat_from_inputs(x_seq, x_ctx, sensor_metadata)
                    st.success("Uploaded CSV successfully parsed!")
                else:
                    st.error("Uploaded CSV must have at least 10 rows and 36 sensor columns.")
            except Exception as e:
                st.error(f"Error parsing CSV: {e}")

    # Run inference and multi-modal calculations
    if x_seq is not None and x_ctx is not None:
        if "TCN" in active_model:
            model_type = "tcn"
        elif "GRU" in active_model:
            model_type = "gru"
        else:
            model_type = "rf"

        result = predictor.forecast_and_classify(
            x_seq=x_seq,
            x_ctx=x_ctx,
            x_flat=x_flat if x_flat is not None else np.zeros(1728),
            model_type=model_type,
            sensor_id=focal_sensor,
            ctx_hour=sel_hour,
        )

        st.markdown("---")
        st.subheader("Multi-Modal Predictive Intelligence (Next 15-Minute Window)")

        # Metric row 1: Traffic Volume & Categories
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        pred_val = result["sensor_prediction"]
        cat_label = result["sensor_category"]
        low_th, high_th = result["thresholds"]

        with col_m1:
            st.metric(
                label=f"Predicted Volume (Sensor #{focal_sensor:02d})",
                value=f"{pred_val:.4f}",
                delta=f"{(pred_val - y_actual[focal_sensor]):.4f}" if y_actual is not None else None,
                delta_color="inverse",
            )
        with col_m2:
            st.metric(
                label="Ground Truth Volume",
                value=f"{y_actual[focal_sensor]:.4f}" if y_actual is not None else "N/A (Simulated Input)",
            )
        with col_m3:
            badge_class = "band-low" if cat_label == "Low Volume" else ("band-mod" if cat_label == "Moderate Volume" else "band-high")
            st.markdown(
                f"<div style='margin-top: 8px;'><strong>Congestion Band:</strong><br>"
                f"<span class='{badge_class}' style='font-size: 1.15rem;'>{cat_label}</span></div>",
                unsafe_allow_html=True,
            )
        with col_m4:
            st.markdown(
                f"<div style='margin-top: 8px;'><strong>Inference Architecture:</strong><br>"
                f"<code>{active_model}</code></div>",
                unsafe_allow_html=True,
            )

        # Metric row 2: Multi-Modal Urban & Clean Energy Indicators
        st.markdown("#### 🌿 Environmental & Infrastructure Impact Indicators")
        col_e1, col_e2, col_e3, col_e4 = st.columns(4)

        with col_e1:
            st.metric(
                label="Sensor Section CO₂ Emissions",
                value=f"{result['sensor_co2_kg']:.1f} kg",
                help="Estimated carbon emissions based on speed-idling non-linearities in this 15-minute window.",
            )
        with col_e2:
            st.metric(
                label="Corridor Total CO₂ (36 Sensors)",
                value=f"{result['total_co2_kg']:.1f} kg",
                help="Total emissions across the entire monitored corridor.",
            )
        with col_e3:
            st.metric(
                label="EV Fast-Charging Surge Index",
                value=f"{result['sensor_ev_surge_index']:.1f}%",
                help="Surge pressure on local electrical substations & fast-charging stations.",
            )
        with col_e4:
            metering = result["metering_active"]
            advisory_speed = result["recommended_speed_kmh"]
            st.metric(
                label="Dynamic Advisory Speed",
                value=f"{advisory_speed:.0f} km/h",
                delta="Ramp Metering Active" if metering else "Free Flow",
                delta_color="inverse" if metering else "normal",
            )

        # Corridor Bar Charts
        st.markdown("#### 🗺️ Spatio-Temporal Corridor Forecast (36 Sensors)")
        preds_all = result["all_predictions"]
        speeds_all = result["all_recommended_speeds"]
        co2_all = compute_carbon_emissions(preds_all)
        surge_all = compute_ev_grid_surge_index(preds_all, ctx_hour=sel_hour)

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.write("**Forecasted Traffic Volume across Sensors:**")
            df_traffic = pd.DataFrame({
                "Sensor": [f"S{i:02d}" for i in range(36)],
                "Volume": preds_all,
            }).set_index("Sensor")
            st.bar_chart(df_traffic["Volume"], height=250)

        with col_g2:
            st.write("**EV Charging & Smart Grid Surge Pressure Index (%):**")
            df_surge = pd.DataFrame({
                "Sensor": [f"S{i:02d}" for i in range(36)],
                "Surge Pressure (%)": surge_all,
            }).set_index("Sensor")
            st.bar_chart(df_surge["Surge Pressure (%)"], height=250)

        # Historical Trend Plot
        st.markdown(f"#### Focal Sensor #{focal_sensor:02d}: Historical Sequence & Attention-Weighted Lookahead")
        fig, ax = plt.subplots(figsize=(10, 3.2), dpi=200)
        hist_steps = np.arange(-9, 1)
        hist_vals = x_seq[:, focal_sensor]

        ax.plot(hist_steps, hist_vals, marker="o", color="#1565c0", lw=2, label="Historical Volume (15m Steps)")
        ax.axhline(low_th, color="#2e7d32", linestyle=":", alpha=0.7, label=f"Low Volume Limit ({low_th:.3f})")
        ax.axhline(high_th, color="#c62828", linestyle=":", alpha=0.7, label=f"High Volume Threshold ({high_th:.3f})")

        ax.scatter([1], [pred_val], color="#f57c00", s=110, zorder=5, label=f"Predicted t+1 ({pred_val:.3f})")
        if y_actual is not None:
            ax.scatter([1], [y_actual[focal_sensor]], color="#2e7d32", s=110, marker="x", zorder=5, label=f"Ground Truth t+1 ({y_actual[focal_sensor]:.3f})")

        ax.set_xlabel("Time Step Relative to Present (t)")
        ax.set_ylabel("Normalized Volume")
        ax.set_ylim(-0.02, 1.05)
        ax.legend(loc="upper left", frameon=True, fontsize=8)
        st.pyplot(fig)


def render_model_comparison(metrics_data):
    st.markdown('<div class="main-title">Rigorous Experimental Benchmark: 3-Model Comparison</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Evaluation on Held-Out Test Horizon (N = 840 Continuous Quarter-Hours / 30,240 Total Sensor Predictions)</div>',
        unsafe_allow_html=True,
    )

    if not metrics_data:
        st.warning("No metrics summary found. Please run training pipeline first.")
        return

    rf_data = metrics_data.get("baseline_rf", {})
    gru_data = metrics_data.get("flowcast_gru", {})
    tcn_data = metrics_data.get("flowcast_tcn_attention", {})

    # Comparison Table
    st.markdown("### 📊 Continuous Regression & Categorization Performance")
    comp_table = [
        {
            "Architecture": "Random Forest Baseline",
            "Model Type": "Ensemble Trees",
            "MAE ↓": f"{rf_data['regression']['mae']:.4f}",
            "RMSE ↓": f"{rf_data['regression']['rmse']:.4f}",
            "R² Score ↑": f"{rf_data['regression']['r2']:.4f}",
            "Macro-F1 ↑": f"{rf_data['classification']['macro_f1']:.4f}",
            "Accuracy ↑": f"{rf_data['classification']['accuracy'] * 100:.2f}%",
            "Training Time": f"{metrics_data.get('train_time_rf_sec', 0):.2f}s",
        },
        {
            "Architecture": "FlowCast GRU",
            "Model Type": "Recurrent + MLP Fusion",
            "MAE ↓": f"{gru_data['regression']['mae']:.4f}",
            "RMSE ↓": f"{gru_data['regression']['rmse']:.4f}",
            "R² Score ↑": f"{gru_data['regression']['r2']:.4f}",
            "Macro-F1 ↑": f"{gru_data['classification']['macro_f1']:.4f}",
            "Accuracy ↑": f"{gru_data['classification']['accuracy'] * 100:.2f}%",
            "Training Time": f"{metrics_data.get('train_time_gru_sec', 0):.2f}s",
        },
        {
            "Architecture": "FlowCast TCN-Attention (Novel)",
            "Model Type": "Dilated Causal CNN + Multi-Head Self-Attention",
            "MAE ↓": f"{tcn_data['regression']['mae']:.4f}",
            "RMSE ↓": f"{tcn_data['regression']['rmse']:.4f}",
            "R² Score ↑": f"{tcn_data['regression']['r2']:.4f}",
            "Macro-F1 ↑": f"{tcn_data['classification']['macro_f1']:.4f}",
            "Accuracy ↑": f"{tcn_data['classification']['accuracy'] * 100:.2f}%",
            "Training Time": f"{metrics_data.get('train_time_tcn_sec', 0):.2f}s",
        },
    ]
    st.dataframe(pd.DataFrame(comp_table).set_index("Architecture"), use_container_width=True)

    st.markdown("---")
    st.subheader("Publication-Quality Experimental Figures")

    tab_bars, tab_curves, tab_preds, tab_cm = st.tabs([
        "📊 Three-Model Benchmark Comparison",
        "📈 Training Loss Curves",
        "🎯 Prediction Fidelity (Actual vs Predicted)",
        "🔲 Confusion Matrices",
    ])

    fig_dir = Path("artifacts/figures")
    with tab_bars:
        comp_img = fig_dir / "model_comparison.png"
        if comp_img.exists():
            st.image(str(comp_img), caption="Comparative Error & Classification Metrics across All 3 Models", use_container_width=True)

    with tab_curves:
        loss_img = fig_dir / "loss_curves.png"
        if loss_img.exists():
            st.image(str(loss_img), caption="FlowCast TCN-Attention: Training vs Validation Loss Curve (Early Stopping)", use_container_width=True)

    with tab_preds:
        pred_img = fig_dir / "actual_vs_predicted_sensor0.png"
        if pred_img.exists():
            st.image(str(pred_img), caption="Focal Sensor Prediction Fidelity & Scatter vs Ideal y=x", use_container_width=True)

    with tab_cm:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            cm_tcn = fig_dir / "confusion_matrix_tcn.png"
            if cm_tcn.exists():
                st.image(str(cm_tcn), caption="FlowCast TCN-Attention Confusion Matrix", use_container_width=True)
        with col_c2:
            cm_rf = fig_dir / "confusion_matrix_baseline.png"
            if cm_rf.exists():
                st.image(str(cm_rf), caption="Random Forest Baseline Confusion Matrix", use_container_width=True)


def render_architecture_and_audit(metrics_data):
    st.markdown('<div class="main-title">System Architecture, Methodology & Dataset Audit</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Detailed Mathematical Formulation, Causality Protocols & Project Attribution</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("### 🧠 The FlowCast TCN-Attention Architecture")
        st.markdown(
            """
            Unlike basic RNNs that suffer from sequential bottlenecks and exploding/vanishing gradients, FlowCast TCN-Attention fuses three advanced paradigms:
            1. **Dilated Causal 1D Convolutions:** Applies left-padded causal kernels with exponential dilation factors ($d = 2^0, 2^1$) ensuring strict temporal causality:
               $$\\text{Output}_t = f(\\text{Input}_{\\le t})$$
            2. **Multi-Head Temporal Self-Attention:** Employs 4-head scaled dot-product attention across all historical quarter-hours to learn dynamic inter-timestep weights:
               $$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$
            3. **Contextual Day/Hour Embedding:** A parallel 2-layer MLP transforms 29 one-hot calendar flags (5 days + 24 hours), which are concatenated before the bounded multi-sensor output head.
            """
        )

        st.markdown("### ⚡ Multi-Modal Domain Innovations")
        st.markdown(
            """
            - **Carbon Footprint Modeling:** Uses EPA non-linear vehicle emission equations factoring in queue delay and idling fuel penalties.
            - **EV Fast-Charging Surge Index:** Maps traffic flow directly into corridor grid charging demand surges to prevent transformer overload.
            - **Dynamic Speed Harmonization:** Implements real-time variable speed limit (VSL) advisories to smooth bottleneck shockwaves.
            """
        )

    with col2:
        st.markdown("### 📋 Dataset Metadata & Audit")
        st.markdown(
            """
            - **Dataset Name**: UCI Traffic Flow Forecasting (ID: 734)
            - **DOI / Citation**: 10.1145/3339823 (Zhao et al., ACM TSAS)
            - **Sensor Network**: 36 highway loop detector stations
            - **Features**: 48 per station (10 historical + 29 time context + 9 static road attributes)
            - **Total Records**: 2,101 continuous quarter-hours (75,636 individual readings)
            - **Missing Values**: 0 (100% clean)
            - **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
            - **Data Splits**: 1,061 train / 200 validation / 840 held-out test
            """
        )


if __name__ == "__main__":
    main()
