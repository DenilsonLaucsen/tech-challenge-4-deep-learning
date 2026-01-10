from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from torch.utils.data import DataLoader, TensorDataset

from src.data.scaler import TimeSeriesScaler

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)


class DataStrategy(ABC):
    """Interface abstrata para estratégias de processamento de dados de séries temporais."""

    @abstractmethod
    def process(
        self,
        tickers: list[str],
        period: Optional[str],
        start: Optional[str],
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Processa dados de tickers e retorna tensores X (sequências) e y (targets)."""

    @staticmethod
    def load_data(
        tickers: list[str],
        period: Optional[str] = None,
        start: Optional[str] = None,
        force_download: bool = False,
    ) -> pd.DataFrame:
        if not tickers:
            raise ValueError("Lista de tickers não pode estar vazia")

        dfs = []
        for ticker in tickers:
            local_path = DATA_DIR / f"{ticker}.csv"

            # Tentar carregar do cache local
            if local_path.exists() and not force_download:
                try:
                    df = pd.read_csv(
                        local_path,
                        index_col=0,
                        parse_dates=True,
                    )
                except Exception as e:
                    print(f"[WARNING] Erro ao ler cache de {ticker}: {e}. Baixando novamente...")
                    df = None
            else:
                df = None

            # Baixar dados do yfinance se necessário
            if df is None:
                try:
                    ticker_obj = yf.Ticker(ticker)
                    if period:
                        df = ticker_obj.history(period=period)
                    elif start:
                        df = ticker_obj.history(start=start)
                    else:
                        df = ticker_obj.history(period="max")

                    if df.empty:
                        raise ValueError(f"yfinance retornou DataFrame vazio para ticker '{ticker}'")

                    df.to_csv(local_path)
                except Exception as e:
                    raise RuntimeError(f"Erro ao baixar dados para ticker '{ticker}': {e}")

            # Garantir que colunas essenciais existem
            required_columns = ["High", "Low", "Close", "Volume"]
            missing = [c for c in required_columns if c not in df.columns]
            if missing:
                raise ValueError(
                    f"Ticker '{ticker}' faltando colunas: {missing} "
                    f"(encontradas: {list(df.columns)})"
                )

            # Converter colunas para numérico (NaN para não-conversíveis)
            df_selected = df[required_columns].copy()
            df_selected = df_selected.apply(
                lambda col: pd.to_numeric(col, errors="coerce")
            )

            # Remover linhas com valores ausentes
            n_rows_before = len(df_selected)
            df_selected = df_selected.dropna()
            n_rows_after = len(df_selected)

            if df_selected.empty:
                raise ValueError(
                    f"DataFrame do ticker '{ticker}' ficou vazio após conversão "
                    f"numérica e remoção de NaNs ({n_rows_before} → {n_rows_after} linhas). "
                    f"Verifique os dados brutos."
                )

            # Renomear colunas com prefixo do ticker
            df_renamed = df_selected.rename(
                columns=lambda col: f"{ticker}_{col}"
            )
            dfs.append(df_renamed)

        # Juntar por índice temporal (inner join) e remover linhas com NaNs
        df_result = pd.concat(dfs, axis=1, join='inner').dropna()
        if df_result.empty:
            raise ValueError(
                "DataFrame resultante ficou vazio após concatenação e remoção de NaNs. "
                "Verifique se tickers, período e dados estão consistentes."
            )
        return df_result

    @staticmethod
    def create_sequences(
        data: np.ndarray,
        seq_len: int,
        target_col_idx: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Converter para array numérico
        try:
            data_np = np.asarray(data, dtype=np.float32)
        except (ValueError, TypeError):
            # Fallback: converter elemento a elemento
            data_np = np.array(
                [[float(x) for x in row] for row in data], dtype=np.float32
            )

        n_timesteps = data_np.shape[0]
        if n_timesteps <= seq_len:
            raise ValueError(
                f"Dados insuficientes para criar sequências: "
                f"{n_timesteps} timesteps ≤ {seq_len} seq_len. "
                f"Necessário pelo menos {seq_len + 1} timesteps."
            )

        # Criar sequências com sliding window
        X_sequences = []
        y_values = []

        for i in range(n_timesteps - seq_len):
            # Extrair janela de entrada
            X_sequences.append(data_np[i : i + seq_len, :])
            # Target: valor da coluna especificada no próximo timestep
            y_values.append(data_np[i + seq_len, target_col_idx])

        
        # Converter para tensores PyTorch
        X = torch.from_numpy(np.stack(X_sequences))  # Shape: (N, seq_len, features)
        y = torch.from_numpy(np.array(y_values, dtype=np.float32)).reshape(-1, 1)  # Shape: (N, 1)

        return X, y


class NoProcessingSingle(DataStrategy):
    """Estratégia sem processamento extra: usa apenas um ticker com features básicas."""

    def process(
        self,
        tickers: list[str],
        period: Optional[str],
        start: Optional[str],
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not tickers:
            raise ValueError("NoProcessingSingle requer ao menos 1 ticker")
        
        df = self.load_data([tickers[0]], period=period, start=start)
        # Ordem de colunas: [TICKER_High, TICKER_Low, TICKER_Close, TICKER_Volume]
        arr = df.values
        return self.create_sequences(arr, seq_len)


class NoProcessingMultiple(DataStrategy):
    """Estratégia sem processamento extra: usa múltiplos tickers com features básicas."""

    def process(
        self,
        tickers: list[str],
        period: Optional[str],
        start: Optional[str],
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(tickers) < 2:
            raise ValueError("NoProcessingMultiple requer dois ou mais tickers")
        
        df = self.load_data(tickers, period=period, start=start)
        arr = df.values
        return self.create_sequences(arr, seq_len)


class RangeSingle(DataStrategy):
    """Estratégia com feature extra: adiciona Range (High - Low) para um ticker."""

    def process(
        self,
        tickers: list[str],
        period: Optional[str],
        start: Optional[str],
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not tickers:
            raise ValueError("RangeSingle requer ao menos 1 ticker")
        
        ticker = tickers[0]
        df = self.load_data([ticker], period=period, start=start)
        
        # Adicionar feature: Range = High - Low
        df[f"{ticker}_Range"] = df[f"{ticker}_High"] - df[f"{ticker}_Low"]
        df = df.dropna()
        
        if df.empty:
            raise ValueError(f"DataFrame vazio após calcular Range para ticker '{ticker}'")
        
        arr = df.values
        return self.create_sequences(arr, seq_len)


class RangeMultiple(DataStrategy):
    """Estratégia com feature extra: adiciona Range (High - Low) para cada ticker."""

    def process(
        self,
        tickers: list[str],
        period: Optional[str],
        start: Optional[str],
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(tickers) < 2:
            raise ValueError("RangeMultiple requer dois ou mais tickers")
        
        df = self.load_data(tickers, period=period, start=start)
        
        # Adicionar feature Range para cada ticker
        for ticker in tickers:
            df[f"{ticker}_Range"] = df[f"{ticker}_High"] - df[f"{ticker}_Low"]
        
        df = df.dropna()
        
        if df.empty:
            raise ValueError("DataFrame vazio após calcular Range para todos os tickers")
        
        arr = df.values
        return self.create_sequences(arr, seq_len)


class DataPipeline:
    """
    Pipeline de dados:
    - Executa a strategy (gera X_raw, y_raw como tensores)
    - Realiza split temporal train/val/test
    - Ajusta scaler APENAS no conjunto de treino (flattened windows)
    - Transforma X e y em cada split
    - Retorna (train_loader, val_loader, test_loader)
    """

    def __init__(
        self,
        strategy: DataStrategy,
        batch_size: int = 32,
        shuffle: bool = True,
        scaler: Optional[TimeSeriesScaler] = None,
        scaler_name: Optional[str] = None,
        enable_mlflow: bool = True,
        mlflow_logger: Optional[MLFlowLogger] = None,
        is_ray_run: bool = False,
    ):
        self.strategy = strategy
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.scaler = scaler or TimeSeriesScaler()
        self.scaler_name = scaler_name
        self.enable_mlflow = enable_mlflow
        self.mlflow_logger = mlflow_logger
        self.is_ray_run = is_ray_run

    def _log_mlflow_params(
        self,
        *,
        tickers,
        period,
        start,
        seq_len,
        train_ratio,
        val_ratio,
        n_samples,
        n_features,
    ):
        """Loga apenas metadados de dados."""
        if self.enable_mlflow and self.mlflow_logger:
            exp = self.mlflow_logger.experiment
            run_id = self.mlflow_logger.run_id
            exp.log_param(run_id, "tickers", ",".join(tickers))
            exp.log_param(run_id, "period", period)
            exp.log_param(run_id, "start_date", start)
            exp.log_param(run_id, "seq_len", seq_len)
            exp.log_param(run_id, "train_ratio", train_ratio)
            exp.log_param(run_id, "val_ratio", val_ratio)
            exp.log_param(run_id, "batch_size", self.batch_size)
            exp.log_param(run_id, "shuffle", self.shuffle)
            exp.log_param(run_id, "n_samples", n_samples)
            exp.log_param(run_id, "n_features", n_features)
            exp.log_param(run_id, "scaler", self.scaler.__class__.__name__)

    def run(
        self,
        tickers: list[str],
        period: Optional[str] = None,
        start: Optional[str] = None,
        seq_len: int = 30,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:

        # 1) gerar sequências brutas
        X_raw, y_raw = self.strategy.process(
            tickers, period=period, start=start, seq_len=seq_len
        )

        n_samples, seq_length, n_features = X_raw.shape

        # 🔹 Expor número de features para o TrainerContext
        self.num_features = n_features

        # 🔹 MLflow — log inicial
        if self.enable_mlflow and self.mlflow_logger:
            self._log_mlflow_params(
                tickers=tickers,
                period=period,
                start=start,
                seq_len=seq_len,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                n_samples=n_samples,
                n_features=n_features,
            )

        # ---- validações (mantidas) ----
        if n_samples < 3:
            raise ValueError("Número de amostras muito pequeno para split.")

        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        train_end = max(train_end, 1)
        val_end = max(val_end, train_end + 1)
        val_end = min(val_end, n_samples - 1)

        # 3) preparar numpy arrays
        X_np = X_raw.numpy()
        y_np = y_raw.numpy().reshape(-1)

        # 4) ajustar scaler somente no treino
        X_train_flat = X_np[:train_end].reshape(-1, n_features)
        self.scaler.fit_transform(X_train_flat)

        # 5) transformar todos os dados
        X_all_scaled = self.scaler.transform(
            X_np.reshape(-1, n_features)
        ).reshape(n_samples, seq_length, n_features)

        target_idx = 2
        def scale_y_array(y_array: np.ndarray) -> np.ndarray:
            dummy = np.zeros((len(y_array), n_features), dtype=np.float32)
            dummy[:, target_idx] = y_array.reshape(-1)
            scaled = self.scaler.transform(dummy)
            return scaled[:, target_idx].reshape(-1, 1)

        y_all_scaled = scale_y_array(y_np)

        # 6) splits → tensores
        X_train = torch.tensor(X_all_scaled[:train_end], dtype=torch.float32)
        y_train = torch.tensor(y_all_scaled[:train_end], dtype=torch.float32)
        X_val = torch.tensor(X_all_scaled[train_end:val_end], dtype=torch.float32)
        y_val = torch.tensor(y_all_scaled[train_end:val_end], dtype=torch.float32)
        X_test = torch.tensor(X_all_scaled[val_end:], dtype=torch.float32)
        y_test = torch.tensor(y_all_scaled[val_end:], dtype=torch.float32)

        # 7) salvar scaler + log como artefato
        if self.scaler_name and self.enable_mlflow and self.mlflow_logger:
            if self.is_ray_run:
                self.scaler.save(self.scaler_name, save_local=False)
            else:
                path = self.scaler.save(self.scaler_name)
                self.mlflow_logger.experiment.log_artifact(
                    self.mlflow_logger.run_id,
                    path
                )

        # 8) dataloaders
        return (
            DataLoader(TensorDataset(X_train, y_train), batch_size=self.batch_size, shuffle=self.shuffle),
            DataLoader(TensorDataset(X_val, y_val), batch_size=self.batch_size, shuffle=False),
            DataLoader(TensorDataset(X_test, y_test), batch_size=self.batch_size, shuffle=False),
        )
