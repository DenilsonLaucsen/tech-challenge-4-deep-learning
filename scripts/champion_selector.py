"""
Champion Selector Script

Responsibility:
- Query MLflow experiment runs
- Select the best run based on val_rmse (minimization)
- Reconstruct the champion configuration
- Persist configs/best_config.yaml
"""

from pathlib import Path
from datetime import datetime
import yaml
import ast


import mlflow
from mlflow.tracking import MlflowClient


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
EXPERIMENT_NAME = "lstm_strategy_experiments"
PRIMARY_METRIC = "val_rmse"
OUTPUT_CONFIG_PATH = Path("configs/best_config.yaml")


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def get_best_run(client: MlflowClient, experiment_id: str):
    """
    Retrieve the best MLflow run based on PRIMARY_METRIC.
    """
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=[f"metrics.{PRIMARY_METRIC} ASC"],
        max_results=1,
    )

    if not runs:
        raise RuntimeError("No MLflow runs found for experiment.")

    return runs[0]


def build_champion_config(run):
    """
    Build a semantic champion configuration from the MLflow run.
    """

    params = run.data.params
    metrics = run.data.metrics

    # -----------------------------------------------------------------
    # IMPORTANT:
    # Adjust param names here if your MLflow params differ
    # -----------------------------------------------------------------

    print(params)
    champion_config = {
        "metadata": {
            "selected_on": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_name": EXPERIMENT_NAME,
            "run_id": run.info.run_id,
            "metric": PRIMARY_METRIC,
            "metric_value": metrics.get(PRIMARY_METRIC),
            "strategy": run.info.run_name,
        },
        "model": {
            "type": "LSTM",
            "input_size": int(params.get("input_size", 4)),
            "hidden_size": int(params["hidden_size"]),
            "num_layers": int(params["num_layers"]),
            "dropout": float(params.get("dropout", 0.0)),
            "output_size": int(params.get("output_size", 1)),
            "layer_config": ast.literal_eval(params["layer_config"]),
        },
        "training": {
            "learning_rate": float(params["learning_rate"]),
            "batch_size": int(params["batch_size"]),
            "num_epochs": int(params["num_epochs"]),
            "shuffle": params["shuffle"].lower() == "true",
        },
        "data": {
            "tickers": yaml.safe_load(params["tickers"])
            if isinstance(params.get("tickers"), str)
            else params.get("tickers"),
            "period": params["period"],
            "seq_len": int(params["seq_len"]),
            "train_ratio": float(params["train_ratio"]),
            "val_ratio": float(params["val_ratio"]),
            "scaler": params.get("scaler_name", "TimeSeriesScaler"),
        },
    }

    return champion_config


def save_yaml(config: dict, output_path: Path):
    """
    Save YAML config to disk.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)

    print(f"🏆 Champion config saved at: {output_path}")


def log_artifact_to_mlflow(run_id: str, artifact_path: Path):
    """
    Log the champion config as an MLflow artifact.
    """
    mlflow.start_run(run_id=run_id)
    mlflow.log_artifact(str(artifact_path), artifact_path="champion")
    mlflow.end_run()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print("🔍 Selecting champion model from MLflow...")

    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found.")

    best_run = get_best_run(client, experiment.experiment_id)

    print(f"🏆 Best run found: {best_run.info.run_id}")
    print(f"📉 {PRIMARY_METRIC}: {best_run.data.metrics.get(PRIMARY_METRIC)}")

    champion_config = build_champion_config(best_run)

    save_yaml(champion_config, OUTPUT_CONFIG_PATH)

    # Log YAML as artifact linked to the champion run
    log_artifact_to_mlflow(best_run.info.run_id, OUTPUT_CONFIG_PATH)

    print("✅ Champion selection completed successfully.")


if __name__ == "__main__":
    main()
