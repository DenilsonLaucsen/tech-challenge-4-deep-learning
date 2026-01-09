from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

SCALER_DIR = Path("artifacts/scalers")
SCALER_DIR.mkdir(parents=True, exist_ok=True)


class TimeSeriesScaler:
    def __init__(self):
        """Inicializa scaler."""
        self.scaler = StandardScaler()
        self.fitted = False

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        scaled = self.scaler.fit_transform(data)
        self.fitted = True
        return scaled

    def transform(self, data: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Scaler não foi ajustado (fit) antes de transform")
        return self.scaler.transform(data)

    def inverse_transform_target(
        self, y_scaled: np.ndarray, target_idx: int = 2
    ) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Scaler não foi ajustado (fit) antes de inverse_transform")

        # Criar dummy com todas as features para usar inverse_transform
        dummy = np.zeros((len(y_scaled), self.scaler.mean_.shape[0]))
        dummy[:, target_idx] = y_scaled.squeeze()
        inverted = self.scaler.inverse_transform(dummy)
        return inverted[:, target_idx]

    def save(self, name: str, save_local: bool = True) -> Path:
        path = SCALER_DIR / f"{name}.pkl"

        if save_local:
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.scaler, path)

        return path

    def load(self, path: Path) -> None:
        self.scaler = joblib.load(path)
        self.fitted = True
