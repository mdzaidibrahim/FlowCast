"""FlowCast: Deep Learning-Based Short-Horizon Traffic Flow Forecasting.

Streamlit interactive dashboard for academic presentation, live forecasting,
and model comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.data.preprocess import build_x_flat_from_inputs, classify_traffic
from src.models.predict import FlowCastPredictor

# Set page configuration
st.set_page_config(
    page_title="FlowCast | Traffic Forecasting & Congestion Analysis",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished academic appearance
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3d59;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #555555;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #1e3d59;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .band-low {
        color: #2e7d32;
        font-weight: bold;
        background-color: #e8f5e9;
        padding: 4px 8px;
        border-radius: 4px;
    }
    .band-mod {
        color: #f57f17;
        font-weight: bold;
        background-color: #fffde7;
        padding: 4px 8px;
        border-radius: 4px;
    }
    .band-high {
        color: #c62828;
        font-weight: bold;
        background-color: #ffebee;
        padding: 4px 8px;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictor():
    """Load cached inference engine without retraining."""
    return FlowCastPredictor(
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
        with open(metrics_file, "r") as f:
            metrics_data = json.load(f)

    samples_data = {}
    if samples_file.exists():
        with open(samples_file, "r") as f:
            samples_data = json.load(f)

    sensor_metadata = {}
    if meta_file.exists():
        with open(meta_file, "r") as f:
            sensor_metadata = json.load(f)
    else:
        # Fallback to dataset extraction if json does not exist
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

    # Sidebar controls
    st.sidebar.image(
        "https://images.unsplash.com/photo-1506521781263-d8422e82f27a?auto=format&fit=crop&w=400&q=80",
        caption="FlowCast Spatio-Temporal Forecasting",
        use_container_width=True,
    )
    st.sidebar.title("Navigation & Parameters")
    nav_option = st.sidebar.radio(
        "Select Screen",
        ["Live Forecasting Dashboard", "Model Comparison & Diagnostics", "Project Architecture & Audit"],
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Forecasting Engine Settings")
    active_model = st.sidebar.selectbox("Active Inference Model", ["FlowCast GRU", "Random Forest Baseline"])
    focal_sensor = st.sidebar.slider("Focal Sensor Station", min_value=0, max_value=35, value=0, step=1)

    low_th, high_th = predictor.thresholds
    st.sidebar.caption(
        rf"**Category Thresholds (Train Derived):**" "\n"
        rf"- Low Band: $\le {low_th:.4f}$" "\n"
        rf"- Moderate Band: $({low_th:.4f}, {high_th:.4f}]$" "\n"
        rf"- High Band: $> {high_th:.4f}$"
    )

    if nav_option == "Live Forecasting Dashboard":
        render_forecasting_dashboard(predictor, samples_data, sensor_metadata, active_model, focal_sensor)
    elif nav_option == "Model Comparison & Diagnostics":
        render_model_comparison(metrics_data)
    else:
        render_architecture_and_audit(metrics_data)


def render_forecasting_dashboard(predictor, samples_data, sensor_metadata, active_model, focal_sensor):
    st.markdown('<div class="main-title">FlowCast: Short-Horizon Traffic Forecasting</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Multi-Output 15-Minute Urban Arterial Volume Prediction & Congestion Band Classification</div>',
        unsafe_allow_html=True,
    )

    # Concise How to Use Guide
    with st.expander("ℹ️ How to Use FlowCast & System Scope", expanded=False):
        st.markdown(
            """
            **System Overview & Purpose:**
            - **Prototype Scope:** FlowCast is a reproducible machine learning benchmark developed on historical California highway loop detector data (UCI Traffic Flow Forecasting Dataset #734). It is an offline research prototype, not a real-time municipal traffic signal controller or a model validated on Indian arterial networks.
            - **15-Minute Lookahead Prediction:** Given 10 consecutive historical observations (2.5 hours at 15-minute intervals) and calendar context, FlowCast simultaneously forecasts traffic volume for all 36 network sensors 15 minutes into the future ($t+1$).
            - **Volume Category Bands:** Continuous predictions in $[0.0, 1.0]$ are translated into three operational bands (**Low**, **Moderate**, **High**) using empirical tertile boundaries derived strictly from the training split.
            - **Demonstration Ground Truth:** In *Bundled Demonstration Mode*, ground truth ($y_{true}$) is loaded from held-out test steps to show exact prediction error ($\Delta$). In *Custom Scenario* and *CSV Input*, predictions reflect synthetic or user-supplied sequences without pre-existing ground truth.
            - **Model Comparison:** Compare the deep learning model (**FlowCast GRU** with contextual MLP) against the competitive **Random Forest Baseline** using the sidebar screen selector.
            """
        )

    # Input method selection
    tab_demo, tab_custom, tab_upload = st.tabs(["🚀 Bundled Demonstration Mode", "⚙️ Custom Scenario Builder", "📂 CSV Batch Input"])

    x_seq, x_ctx, x_flat, y_actual = None, None, None, None

    with tab_demo:
        st.write("Test the system instantly using real time-series instances from the held-out test horizon.")
        sample_list = samples_data.get("samples", [])
        if sample_list:
            demo_names = [
                f"Sample #{s['test_index']} (Time Step t+{s['test_index']} in Test Set)"
                for s in sample_list
            ]
            selected_sample_idx = st.selectbox("Select Test Scenario:", range(len(sample_list)), format_func=lambda i: demo_names[i])
            curr_sample = sample_list[selected_sample_idx]
            x_seq = np.array(curr_sample["x_seq"])
            x_ctx = np.array(curr_sample["x_ctx"])
            x_flat = np.array(curr_sample["x_flat"])
            y_actual = np.array(curr_sample["y_true"])
        else:
            st.warning("Demonstration samples file not found. Generating default sample.")
            x_seq = np.full((10, 36), 0.3, dtype=np.float32)
            x_ctx = np.zeros(29, dtype=np.float32)
            x_ctx[0] = 1.0  # Monday
            x_ctx[17] = 1.0  # Hour 12
            x_flat = build_x_flat_from_inputs(x_seq, x_ctx, sensor_metadata)

    with tab_custom:
        st.write("Manually configure contextual time and focal sensor history.")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            sel_day = st.selectbox("Day of Week:", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
            sel_hour = st.slider("Hour of Day (0-23):", 0, 23, 8)
        with col_c2:
            base_vol = st.slider("Base Traffic Volume for Network:", 0.0, 1.0, 0.35, 0.05)
            focal_vol = st.slider(f"Focal Sensor #{focal_sensor} Recent Volume:", 0.0, 1.0, 0.45, 0.05)

        if st.button("Generate Forecast for Custom Scenario"):
            # Construct synthetic inputs
            x_seq = np.full((10, 36), base_vol, dtype=np.float32)
            x_seq[:, focal_sensor] = np.linspace(focal_vol * 0.8, focal_vol, 10)
            x_ctx = np.zeros(29, dtype=np.float32)
            day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4}
            x_ctx[day_map[sel_day]] = 1.0
            x_ctx[5 + sel_hour] = 1.0
            x_flat = build_x_flat_from_inputs(x_seq, x_ctx, sensor_metadata)

    with tab_upload:
        st.write("Upload a CSV file containing historical readings. Format: at least 10 historical values and 36 sensor columns.")
        uploaded_file = st.file_uploader("Upload Input CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Uploaded CSV loaded with shape: {df.shape}")
                if df.shape[1] >= 36 and len(df) >= 10:
                    x_seq = df.iloc[:10, :36].values.astype(np.float32)
                    x_ctx = np.zeros(29, dtype=np.float32)
                    x_ctx[0] = 1.0  # Default Monday
                    x_ctx[13] = 1.0  # Default 08:00 AM
                    x_flat = build_x_flat_from_inputs(x_seq, x_ctx, sensor_metadata)
                else:
                    st.error("Uploaded CSV must have at least 10 rows and 36 sensor columns.")
            except Exception as e:
                st.error(f"Error parsing CSV: {e}")

    # Run inference and display results
    if x_seq is not None and x_ctx is not None:
        model_type = "gru" if active_model == "FlowCast GRU" else "rf"
        result = predictor.forecast_and_classify(
            x_seq=x_seq,
            x_ctx=x_ctx,
            x_flat=x_flat if x_flat is not None else np.zeros(1728),
            model_type=model_type,
            sensor_id=focal_sensor,
        )

        st.markdown("---")
        st.subheader("Forecast Results (15-Minute Lookahead)")

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
                value=f"{y_actual[focal_sensor]:.4f}" if y_actual is not None else "N/A (Custom Input)",
            )
        with col_m3:
            badge_class = "band-low" if cat_label == "Low Volume" else ("band-mod" if cat_label == "Moderate Volume" else "band-high")
            st.markdown(
                f"<div style='margin-top: 8px;'><strong>Traffic Volume Band:</strong><br>"
                f"<span class='{badge_class}' style='font-size: 1.15rem;'>{cat_label}</span></div>",
                unsafe_allow_html=True,
            )
        with col_m4:
            st.markdown(
                f"<div style='margin-top: 8px;'><strong>Inference Engine:</strong><br>"
                f"<code>{active_model}</code> on <code>{result['device']}</code></div>",
                unsafe_allow_html=True,
            )

        # Network-wide bar chart
        st.markdown("#### Spatio-Temporal Highway Network Forecast (All 36 Sensor Locations)")
        preds_all = result["all_predictions"]

        df_preds = pd.DataFrame({
            "Sensor ID": [f"S{i:02d}" for i in range(36)],
            "Forecast Volume": preds_all,
            "Category": result["all_categories"],
        })
        st.bar_chart(df_preds.set_index("Sensor ID")["Forecast Volume"], height=280)

        # Historical trend of focal sensor
        st.markdown(f"#### Focal Sensor #{focal_sensor:02d}: Historical Sequence (t-9 to t) and Next-15m Forecast")
        fig, ax = plt.subplots(figsize=(10, 3.2), dpi=200)
        hist_steps = np.arange(-9, 1)
        hist_vals = x_seq[:, focal_sensor]

        ax.plot(hist_steps, hist_vals, marker="o", color="#1f77b4", label="Historical Sequence (15m intervals)")
        ax.axhline(low_th, color="green", linestyle=":", alpha=0.7, label=f"Low Band Limit ({low_th:.3f})")
        ax.axhline(high_th, color="red", linestyle=":", alpha=0.7, label=f"High Band Threshold ({high_th:.3f})")

        ax.scatter([1], [pred_val], color="#ff7f0e", s=90, zorder=5, label=f"Predicted t+1 ({pred_val:.3f})")
        if y_actual is not None:
            ax.scatter([1], [y_actual[focal_sensor]], color="#2ca02c", s=90, marker="x", zorder=5, label=f"Actual t+1 ({y_actual[focal_sensor]:.3f})")

        ax.set_xlabel("Time Horizon (Quarter-Hours relative to current time t)")
        ax.set_ylabel("Normalized Volume")
        ax.set_ylim(-0.02, 1.05)
        ax.legend(loc="upper left", frameon=True, fontsize=8)
        st.pyplot(fig)


def render_model_comparison(metrics_data):
    st.markdown('<div class="main-title">Baseline vs FlowCast GRU: Experimental Metrics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Rigorous Evaluation on Held-Out Test Horizon (N = 840 Continuous Quarter-Hours)</div>',
        unsafe_allow_html=True,
    )

    if not metrics_data:
        st.warning("No metrics summary found. Please run training pipeline first.")
        return

    rf_data = metrics_data.get("baseline_rf", {})
    gru_data = metrics_data.get("flowcast_gru", {})

    # Comparison metrics table
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("### 📊 Continuous Regression Metrics")
        reg_df = pd.DataFrame([
            {
                "Model": "Random Forest Baseline",
                "MAE ↓": f"{rf_data['regression']['mae']:.4f}",
                "RMSE ↓": f"{rf_data['regression']['rmse']:.4f}",
                "R² Score ↑": f"{rf_data['regression']['r2']:.4f}",
                "Training Time (s)": f"{metrics_data.get('train_time_rf_sec', 0):.2f}s",
            },
            {
                "Model": "FlowCast GRU (Deep Learning)",
                "MAE ↓": f"{gru_data['regression']['mae']:.4f}",
                "RMSE ↓": f"{gru_data['regression']['rmse']:.4f}",
                "R² Score ↑": f"{gru_data['regression']['r2']:.4f}",
                "Training Time (s)": f"{metrics_data.get('train_time_gru_sec', 0):.2f}s",
            },
        ])
        st.dataframe(reg_df.set_index("Model"), use_container_width=True)

    with col_t2:
        st.markdown("### 🎯 Derived Traffic-Volume Band Classification")
        clf_df = pd.DataFrame([
            {
                "Model": "Random Forest Baseline",
                "Macro-F1 ↑": f"{rf_data['classification']['macro_f1']:.4f}",
                "Accuracy ↑": f"{rf_data['classification']['accuracy']:.4f}",
                "Precision": f"{rf_data['classification']['precision']:.4f}",
                "Recall": f"{rf_data['classification']['recall']:.4f}",
            },
            {
                "Model": "FlowCast GRU (Deep Learning)",
                "Macro-F1 ↑": f"{gru_data['classification']['macro_f1']:.4f}",
                "Accuracy ↑": f"{gru_data['classification']['accuracy']:.4f}",
                "Precision": f"{gru_data['classification']['precision']:.4f}",
                "Recall": f"{gru_data['classification']['recall']:.4f}",
            },
        ])
        st.dataframe(clf_df.set_index("Model"), use_container_width=True)

    st.markdown("---")
    st.subheader("Generated Evaluation Plots")

    tab_curves, tab_preds, tab_cm, tab_bars = st.tabs([
        "📈 Training Loss Curves",
        "🎯 Actual vs Predicted",
        "🔲 Confusion Matrices",
        "📊 Side-by-Side Comparison",
    ])

    fig_dir = Path("artifacts/figures")
    with tab_curves:
        loss_img = fig_dir / "loss_curves.png"
        if loss_img.exists():
            st.image(str(loss_img), caption="FlowCast GRU: Training vs Validation Loss across Epochs", use_container_width=True)
        else:
            st.info("Loss curves figure not available.")

    with tab_preds:
        pred_img = fig_dir / "actual_vs_predicted_sensor0.png"
        if pred_img.exists():
            st.image(str(pred_img), caption="Actual vs Predicted Traffic Volume for Focal Station", use_container_width=True)

    with tab_cm:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            cm_gru = fig_dir / "confusion_matrix_gru.png"
            if cm_gru.exists():
                st.image(str(cm_gru), caption="FlowCast GRU Confusion Matrix", use_container_width=True)
        with col_c2:
            cm_rf = fig_dir / "confusion_matrix_baseline.png"
            if cm_rf.exists():
                st.image(str(cm_rf), caption="Random Forest Confusion Matrix", use_container_width=True)

    with tab_bars:
        comp_img = fig_dir / "model_comparison.png"
        if comp_img.exists():
            st.image(str(comp_img), caption="Baseline vs GRU Model Benchmark Comparison", use_container_width=True)


def render_architecture_and_audit(metrics_data):
    st.markdown('<div class="main-title">System Architecture & Dataset Audit</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Scientific Design, Reproducibility Documentation & Attribution</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("### 🏗️ FlowCast Dual-Branch Neural Architecture")
        st.markdown(
            """
            FlowCast is specifically structured to respect physical spatio-temporal causality:
            1. **Sequential Recurrent Branch**: Consumes the genuine 10-step sequence ($L=10$) of past traffic readings across 36 sensor stations. A 2-layer GRU captures temporal momentum without artificial reshaping.
            2. **Contextual Auxiliary Branch**: Processes 29 one-hot temporal indicators (5 weekdays + 24 hours of the day) via a dense MLP.
            3. **Fusion & Multi-Output Head**: Fuses temporal dynamics with temporal context, predicting next-15m traffic flow across all 36 sensors simultaneously with Sigmoid bounded activations.
            """
        )

        st.markdown("### ⚖️ Academic Integrity & Non-Leaking Protocol")
        st.markdown(
            """
            - **Strict Chronological Splitting**: The dataset maintains 1,061 train steps, 200 validation steps, and 840 held-out test steps without temporal shuffle.
            - **Non-Leaking Thresholds**: Low, Moderate, and High volume thresholds were computed strictly from the 33.3% and 66.7% percentiles of training targets ($0.1822$ and $0.3895$).
            - **Honest Academic Scope**: This prototype represents loop sensors in Northern Virginia / Washington D.C., and does not claim deployment on Indian road networks or active signal preemption.
            """
        )

    with col2:
        st.markdown("### 📋 Dataset Metadata & Audit")
        st.markdown(
            """
            - **Dataset**: UCI Traffic Flow Forecasting (ID: 734)
            - **DOI**: 10.1145/3339823
            - **Sensors**: 36 highway loop detector stations
            - **Features**: 48 per station (10 historical + 29 time context + 9 road attributes)
            - **Samples**: 2,101 continuous quarter-hours (75,636 sensor readings)
            - **Missing Values**: 0 (0.00%)
            - **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
            - **Citation**: Zhao et al., ACM Trans. Spatial Algorithms Syst. (2019)
            """
        )


if __name__ == "__main__":
    main()
