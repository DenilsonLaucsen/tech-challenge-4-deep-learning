import logging
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
import yaml

from src.models.lstm import LSTM, LSTMFactory
from src.models.lstm_params import LSTMParams
from src.utils.model_loader import (
    load_config,
    load_model_state_dict,
    load_scaler,
)

logger = logging.getLogger(__name__)

DEFAULT_TARGET_IDX = 2
DEFAULT_CONFIG_PATH = Path("configs/best_config.yaml")
DEFAULT_MODEL_PATH = Path("artifacts/models/model_final.pt")
DEFAULT_SCALER_NAME = "scaler_final"


def _build_model_from_config(config: Dict) -> LSTM:
    """
    Build LSTM model from configuration.

    Args:
        config: Configuration dictionary with model parameters.

    Returns:
        Instantiated LSTM model.
    """
    model_cfg = config.get("model", {})

    lstm_params = LSTMParams(
        input_size=model_cfg.get("input_size", 4),
        hidden_size=model_cfg.get("hidden_size", 128),
        num_layers=model_cfg.get("num_layers", 1),
        output_size=model_cfg.get("output_size", 1),
        dropout=model_cfg.get("dropout", 0.0),
    )

    layer_config = model_cfg.get("layer_config", ["LSTM", "Linear"])
    factory = LSTMFactory(layer_config, lstm_params)

    return factory.create()


def _normalize_input_sequence(
    sequence: Union[Sequence[float], Sequence[Sequence[float]]],
    scaler,
    target_idx: int = DEFAULT_TARGET_IDX,
) -> np.ndarray:
    """
    Normalize and reshape input sequence for model inference.

    Handles both univariate (list of floats) and multivariate (list of lists)
    input formats. Scaler must be already fitted.

    Args:
        sequence: Input sequence (1D or 2D array-like).
        scaler: Fitted TimeSeriesScaler instance.
        target_idx: Index of target feature in multivariate case.

    Returns:
        Normalized array of shape (1, seq_len, n_features) with dtype float32.

    Raises:
        ValueError: If feature dimensions are incompatible with scaler.
    """
    arr = np.asarray(sequence, dtype=np.float32)

    # Convert univariate to 2D format
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    seq_len, provided_features = arr.shape
    scaler_n_features = scaler.scaler.mean_.shape[0]

    if provided_features == scaler_n_features:
        # All features provided - transform directly
        flat = arr.reshape(-1, scaler_n_features)
        scaled = scaler.transform(flat).reshape(seq_len, scaler_n_features)

    elif provided_features == 1:
        # Only target feature provided - fill zeros for other features
        dummy = np.zeros((seq_len, scaler_n_features), dtype=np.float32)
        dummy[:, target_idx] = arr.squeeze()
        flat = dummy.reshape(-1, scaler_n_features)
        scaled = scaler.transform(flat).reshape(seq_len, scaler_n_features)

    else:
        raise ValueError(
            f"Feature count mismatch: provided {provided_features}, "
            f"scaler expects {scaler_n_features}. "
            f"Send either {scaler_n_features} features per timestep or just 1 (target)."
        )

    return scaled.reshape(1, seq_len, scaler_n_features).astype(np.float32)


def predict(
    sequence: Union[Sequence[float], Sequence[Sequence[float]]],
) -> Dict[str, Union[float, Dict]]:
    """
    Run inference on input sequence.

    Args:
        sequence: Input sequence (list of floats or list of lists).
        config_path: Path to model configuration (default: configs/best_config.yaml).
        model_path: Path to model checkpoint (default: artifacts/models/model_final.pt).
        scaler_name: Name of scaler artifact (default: scaler_final).
        target_idx: Index of target feature in multivariate data.
        device: Torch device to run inference on (default: auto-detect).

    Returns:
        Dictionary containing:
        - prediction: Model output in original scale
        - prediction_scaled: Model output in scaled space
        - model_config: Model configuration metadata

    Raises:
        FileNotFoundError: If config or model files not found.
        ValueError: If input format is invalid.
    """
    target_idx = DEFAULT_TARGET_IDX
    config_path = DEFAULT_CONFIG_PATH
    model_path = DEFAULT_MODEL_PATH
    scaler_name = DEFAULT_SCALER_NAME
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load configuration
    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        logger.error(f"Configuration not found: {e}")
        raise

    # Build and load model
    try:
        model = _build_model_from_config(config)
        state_dict = load_model_state_dict(model_path, device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # Load scaler
    try:
        scaler = load_scaler(scaler_name)
        logger.info(f"Scaler loaded: {scaler_name}")
    except Exception as e:
        logger.error(f"Failed to load scaler: {e}")
        raise

    # Prepare input
    try:
        X = _normalize_input_sequence(sequence, scaler, target_idx=target_idx)
    except ValueError as e:
        logger.error(f"Input preparation failed: {e}")
        raise

    # Run inference
    x_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(x_tensor)

    output_np = output.detach().cpu().numpy().reshape(-1)

    # Inverse transform target feature
    pred_scaled = output_np.reshape(-1, 1)
    pred_original = scaler.inverse_transform_target(
        pred_scaled, target_idx=target_idx
    ).reshape(-1)

    return {
        "prediction": float(pred_original[0]),
        "prediction_scaled": float(pred_scaled[0, 0]),
        "model_config": config.get("model", {}),
    }

