"""Model and scaler loading utilities."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

from src.data.scaler import TimeSeriesScaler

logger = logging.getLogger(__name__)

MODEL_DIR = Path("artifacts/models")
SCALER_DIR = Path("artifacts/scalers")


def load_config(path: Path) -> Dict[str, Any]:
    """
    Load configuration file (YAML or JSON).

    Args:
        path: Path to configuration file (.yaml or .json).

    Returns:
        Dictionary with configuration.

    Raises:
        FileNotFoundError: If file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_scaler(scaler_name: str) -> TimeSeriesScaler:
    """
    Load a TimeSeriesScaler from disk.

    Args:
        scaler_name: Name of scaler file (without .pkl extension).

    Returns:
        Loaded TimeSeriesScaler instance.

    Raises:
        FileNotFoundError: If scaler file does not exist.
    """
    scaler_path = SCALER_DIR / f"{scaler_name}.pkl"

    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    scaler = TimeSeriesScaler()
    scaler.load(scaler_path)
    logger.info(f"Scaler loaded: {scaler_name}")
    return scaler


def load_model_state_dict(model_path: Path, device: torch.device) -> Dict[str, Any]:
    """
    Load model state dict from checkpoint file.

    Args:
        model_path: Path to model checkpoint (.pt file).
        device: Device to load model on.

    Returns:
        State dict dictionary.

    Raises:
        FileNotFoundError: If model file does not exist.
        RuntimeError: If checkpoint format is invalid.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location=device)

        # Handle Lightning checkpoint format (has 'state_dict' key)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            logger.info(f"Loaded Lightning checkpoint format from {model_path}")
            return state_dict

        # Handle raw state_dict
        if isinstance(checkpoint, dict):
            logger.info(f"Loaded state_dict from {model_path}")
            return checkpoint

        raise RuntimeError(
            f"Invalid checkpoint format at {model_path}. "
            "Expected dict or Lightning checkpoint."
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}") from e
