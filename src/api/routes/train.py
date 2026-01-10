from pathlib import Path
import yaml

from fastapi import APIRouter, HTTPException

from src.training.model import (
    TrainerContext,
    NoProcessingSingleStrategy,
    NoProcessingMultipleStrategy,
    RangeSingleStrategy,
    RangeMultipleStrategy,
    TrainingParams,
)
from src.models.lstm_params import LSTMParams
from src.data.scaler import TimeSeriesScaler


router = APIRouter()

# ---------------------------------------------------------------------
# Strategy registry (reuso consciente)
# ---------------------------------------------------------------------
STRATEGY_REGISTRY = {
    "NoProcessingSingle": NoProcessingSingleStrategy,
    "NoProcessingMultiple": NoProcessingMultipleStrategy,
    "RangeSingle": RangeSingleStrategy,
    "RangeMultiple": RangeMultipleStrategy,
}


@router.post("/")
def train_model():
    """
    Train model using the champion configuration (best_config.yaml).
    """

    config_path = Path("configs/best_config.yaml")

    if not config_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Champion config not found. Run champion_selector.py first.",
        )

    # -----------------------------------------------------------------
    # Load champion config
    # -----------------------------------------------------------------
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    try:
        # -----------------------------------------------------------------
        # Build LSTM params
        # -----------------------------------------------------------------
        model_cfg = config["model"]

        lstm_params = LSTMParams(
            input_size=model_cfg["input_size"],
            hidden_size=model_cfg["hidden_size"],
            num_layers=model_cfg["num_layers"],
            output_size=model_cfg["output_size"],
            dropout=model_cfg.get("dropout", 0.0),
        )

        # -----------------------------------------------------------------
        # Build TrainingParams
        # -----------------------------------------------------------------
        training_cfg = config["training"]
        data_cfg = config["data"]

        training_params = TrainingParams(
            tickers=data_cfg["tickers"],
            period=data_cfg["period"],
            seq_len=data_cfg["seq_len"],
            num_epochs=training_cfg["num_epochs"],
            learning_rate=training_cfg["learning_rate"],
            batch_size=training_cfg["batch_size"],
            layer_config=model_cfg["layer_config"],
            lstm_params=lstm_params,
            shuffle=training_cfg["shuffle"],
            scaler=TimeSeriesScaler(),
            scaler_name="scaler_final",
            enable_mlflow=True,
            train_ratio=data_cfg["train_ratio"],
            val_ratio=data_cfg["val_ratio"],
            is_ray_run=False,
            experiment_name="endpoint_training",
        )

        # -----------------------------------------------------------------
        # Strategy selection
        # -----------------------------------------------------------------
        strategy_name = config["metadata"]["strategy"]

        strategy_cls = STRATEGY_REGISTRY.get(strategy_name)
        if not strategy_cls:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        strategy = strategy_cls(training_params)

        # -----------------------------------------------------------------
        # Train
        # -----------------------------------------------------------------
        trainer = TrainerContext(strategy)
        model_path = trainer.train()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "status": "training_completed",
        "model_path": str(model_path),
        "strategy": strategy_name,
        "metric": config["metadata"]["metric"],
        "metric_value": config["metadata"]["metric_value"],
    }
