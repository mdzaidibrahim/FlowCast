"""Single-command cross-platform entry point for FlowCast.

Verifies the environment, ensures the dataset and trained models exist,
and launches the Streamlit forecasting application.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Ensure project root is current working directory and on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REQUIRED_ARTIFACTS = [
    PROJECT_ROOT / "artifacts" / "models" / "gru_best.pth",
    PROJECT_ROOT / "artifacts" / "models" / "baseline_rf.joblib",
    PROJECT_ROOT / "artifacts" / "metrics" / "metrics_summary.json",
    PROJECT_ROOT / "artifacts" / "outputs" / "demo_samples.json",
]


def check_and_prepare_dataset() -> Path:
    """Ensure the UCI Traffic Flow Forecasting dataset is present and valid."""
    from src.data.load_data import download_dataset_if_needed

    raw_dir = PROJECT_ROOT / "data" / "raw"
    mat_file = raw_dir / "traffic_dataset.mat"

    if mat_file.exists() and mat_file.stat().st_size > 1_000_000:
        print("[FlowCast] Dataset verified: data/raw/traffic_dataset.mat")
        return mat_file

    print("[FlowCast] Downloading UCI Traffic Flow Forecasting dataset...")
    mat_file = download_dataset_if_needed(raw_dir)
    print(f"[FlowCast] Dataset ready ({mat_file.stat().st_size / (1024*1024):.1f} MB)")
    return mat_file


def check_and_prepare_artifacts(force_retrain: bool = False) -> None:
    """Verify pre-trained weights and metrics exist; run training pipeline if missing."""
    all_exist = all(p.exists() and p.stat().st_size > 0 for p in REQUIRED_ARTIFACTS)

    if all_exist and not force_retrain:
        print("[FlowCast] Verified model weights and evaluation artifacts.")
        return

    print("[FlowCast] Preparing model checkpoints and evaluation artifacts...")
    print("[FlowCast] Running reproducible training pipeline (~10-15 seconds)...")

    from src.models.train import run_pipeline

    run_pipeline()

    # Verify artifacts were created successfully
    for artifact in REQUIRED_ARTIFACTS:
        if not artifact.exists() or artifact.stat().st_size == 0:
            raise RuntimeError(f"Expected artifact was not generated: {artifact}")

    print("[FlowCast] Model training complete and artifacts verified.")


def launch_streamlit_app(port: int = 8501, headless: bool = False) -> int:
    """Launch the Streamlit web application using the current Python interpreter."""
    app_path = PROJECT_ROOT / "app.py"
    if not app_path.exists():
        raise FileNotFoundError(f"Application entry point not found at {app_path}")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
    ]
    if headless:
        cmd.extend(["--server.headless", "true"])

    print(f"[FlowCast] Launching Streamlit interface on port {port}...")
    print("[FlowCast] Press Ctrl+C in this terminal to stop the server.\n")

    try:
        proc = subprocess.run(cmd)
        return proc.returncode
    except KeyboardInterrupt:
        print("\n[FlowCast] Application stopped by user.")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="FlowCast: Traffic Flow Forecasting & Congestion Classification Launcher"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port number for Streamlit application (default: 8501)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Streamlit in headless server mode (no auto-browser opening)",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retraining of baseline and GRU models",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify dataset and model artifacts without starting Streamlit",
    )

    args = parser.parse_args()

    try:
        check_and_prepare_dataset()
        check_and_prepare_artifacts(force_retrain=args.retrain)

        if args.check_only:
            print("[FlowCast] Environment, data, and models are fully ready.")
            return 0

        return launch_streamlit_app(port=args.port, headless=args.headless)
    except Exception as exc:
        print(f"\n[FlowCast Error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
