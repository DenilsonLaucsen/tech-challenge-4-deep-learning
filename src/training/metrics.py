import torch


def mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Mean Absolute Error
    """
    return torch.mean(torch.abs(y_pred - y_true))


def rmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Root Mean Squared Error
    """
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2))


def mape(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Mean Absolute Percentage Error

    eps avoids division by zero.
    """
    return torch.mean(torch.abs((y_true - y_pred) / (y_true + eps))) * 100.0
