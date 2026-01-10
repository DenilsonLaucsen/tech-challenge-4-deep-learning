import torch
import torch.nn as nn
import pytorch_lightning as pl
import tempfile
from pytorch_lightning.loggers import MLFlowLogger
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from pathlib import Path

from src.data.scaler import TimeSeriesScaler
from src.data.data import DataPipeline, NoProcessingSingle, NoProcessingMultiple, RangeSingle, RangeMultiple
from src.models.lstm import LSTMFactory
from src.training.metrics import mae, rmse, mape

@dataclass
class TrainingParams:
    tickers: List[str]
    period: Optional[str]
    seq_len: int
    num_epochs: int
    learning_rate: float
    batch_size: int
    layer_config: dict
    lstm_params: dict
    shuffle: bool = True
    scaler: Optional[TimeSeriesScaler] = None,
    scaler_name: Optional[str] = None,
    enable_mlflow: bool = True
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    is_ray_run: bool = False
    experiment_name: Optional[str] = None

class LSTMLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for training LSTM-based models.
    """

    def __init__(self, model: nn.Module, lr: float):
        super().__init__()

        # Save hyperparameters for reproducibility
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.lr = lr
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)

        loss = self.criterion(preds, y)
        train_mae = mae(preds, y)
        train_rmse = rmse(preds, y)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_mae", train_mae, on_epoch=True)
        self.log("train_rmse", train_rmse, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)

        loss = self.criterion(preds, y)
        val_mae = mae(preds, y)
        val_rmse = rmse(preds, y)
        val_mape = mape(preds, y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_mae", val_mae, on_epoch=True, prog_bar=True)
        self.log("val_rmse", val_rmse, on_epoch=True)
        self.log("val_mape", val_mape, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class TrainingStrategy(ABC):
    """
    Abstract base class for training strategies.

    Responsibilities:
      - Define WHICH data pipeline is used
      - Define WHICH model factory is used
      - Provide training hyperparameters

    Important:
      - input_size is NOT known at construction time
      - input_size is injected later by TrainerContext
    """

    def __init__(self, training_params: TrainingParams):
        self.training_params = training_params

        self.params: Dict[str, Any] = dict(
            tickers=self.training_params.tickers,
            period=self.training_params.period,
            seq_len=self.training_params.seq_len,
            num_epochs=self.training_params.num_epochs,
            learning_rate=self.training_params.learning_rate,
            batch_size=self.training_params.batch_size,
            train_ratio=self.training_params.train_ratio,
            val_ratio=self.training_params.val_ratio,
        )

        self.layer_config = self.training_params.layer_config
        self.lstm_params = self.training_params.lstm_params

        self.pipeline_kwargs: Dict[str, Any] = dict(
            batch_size=self.training_params.batch_size,
            shuffle=self.training_params.shuffle,
            scaler=self.training_params.scaler,
            scaler_name=self.training_params.scaler_name,
            enable_mlflow=self.training_params.enable_mlflow,
        )

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the strategy."""

    @abstractmethod
    def get_data_pipeline(
        self,
        mlflow_logger: Optional[MLFlowLogger] = None,
        is_ray_run: bool = False,
    ) -> Any:
        """Return a fully configured DataPipeline instance."""

    @abstractmethod
    def get_model_factory(self, input_size: int) -> Any:
        """
        Return a configured ModelFactory.

        Args:
            input_size (int): Number of features produced by the pipeline.
        """

    @abstractmethod
    def get_training_params(self) -> Dict[str, Any]:
        """Return training hyperparameters used by TrainerContext."""

    @classmethod
    def is_compatible(cls, params: TrainingParams) -> bool:
        return True
    

class NoProcessingSingleStrategy(TrainingStrategy):
    """No feature engineering, single ticker."""

    def __init__(self, training_params: TrainingParams):
        super().__init__(training_params)
        self._name = "NoProcessingSingle"

    @property
    def name(self) -> str:
        return self._name
    
    @classmethod
    def is_compatible(cls, params: TrainingParams) -> bool:
        """
        This strategy requires two or more tickers.
        """
        return len(params.tickers) == 1

    def get_data_pipeline(self, mlflow_logger: Optional[MLFlowLogger] = None, is_ray_run: bool = False) -> DataPipeline:
        strategy = NoProcessingSingle()
        return DataPipeline(strategy=strategy, mlflow_logger=mlflow_logger, is_ray_run=is_ray_run, **self.pipeline_kwargs)

    def get_model_factory(self, input_size: int) -> LSTMFactory:
        self.lstm_params.input_size = input_size
        return LSTMFactory(self.layer_config, self.lstm_params)

    def get_training_params(self) -> Dict[str, Any]:
        return asdict(self.training_params)


class NoProcessingMultipleStrategy(TrainingStrategy):
    """No feature engineering, multiple tickers."""

    def __init__(self, training_params: TrainingParams):
        super().__init__(training_params)
        self._name = "NoProcessingMultiple"

    @property
    def name(self) -> str:
        return self._name
    
    @classmethod
    def is_compatible(cls, params: TrainingParams) -> bool:
        """
        This strategy requires two or more tickers.
        """
        return len(params.tickers) >= 2

    def get_data_pipeline(self, mlflow_logger: Optional[MLFlowLogger] = None, is_ray_run: bool = False) -> DataPipeline:
        strategy = NoProcessingMultiple()
        return DataPipeline(strategy=strategy, mlflow_logger=mlflow_logger, is_ray_run=is_ray_run, **self.pipeline_kwargs)

    def get_model_factory(self, input_size: int) -> LSTMFactory:
        self.lstm_params.input_size = input_size
        return LSTMFactory(self.layer_config, self.lstm_params)

    def get_training_params(self) -> Dict[str, Any]:
        return asdict(self.training_params)


class RangeSingleStrategy(TrainingStrategy):
    """Range (High-Low) feature added, single ticker."""

    def __init__(self, training_params: TrainingParams):
        super().__init__(training_params)
        self._name = "RangeSingle"

    @property
    def name(self) -> str:
        return self._name
    
    @classmethod
    def is_compatible(cls, params: TrainingParams) -> bool:
        """
        This strategy requires two or more tickers.
        """
        return len(params.tickers) == 1

    def get_data_pipeline(self, mlflow_logger: Optional[MLFlowLogger] = None, is_ray_run: bool = False) -> DataPipeline:
        strategy = RangeSingle()
        return DataPipeline(strategy=strategy, mlflow_logger=mlflow_logger, is_ray_run=is_ray_run, **self.pipeline_kwargs)

    def get_model_factory(self, input_size: int) -> LSTMFactory:
        self.lstm_params.input_size = input_size
        return LSTMFactory(self.layer_config, self.lstm_params)

    def get_training_params(self) -> Dict[str, Any]:
        return asdict(self.training_params)


class RangeMultipleStrategy(TrainingStrategy):
    """Range (High-Low) feature added, multiple tickers."""

    def __init__(self, training_params: TrainingParams):
        super().__init__(training_params)
        self._name = "RangeMultiple"

    @property
    def name(self) -> str:
        return self._name
    
    @classmethod
    def is_compatible(cls, params: TrainingParams) -> bool:
        """
        This strategy requires two or more tickers.
        """
        return len(params.tickers) >= 2

    def get_data_pipeline(self, mlflow_logger: Optional[MLFlowLogger] = None, is_ray_run: bool = False) -> DataPipeline:
        strategy = RangeMultiple()
        return DataPipeline(strategy=strategy, mlflow_logger=mlflow_logger, is_ray_run=is_ray_run, **self.pipeline_kwargs)

    def get_model_factory(self, input_size: int) -> LSTMFactory:
        self.lstm_params.input_size = input_size
        return LSTMFactory(self.layer_config, self.lstm_params)

    def get_training_params(self) -> Dict[str, Any]:
        return asdict(self.training_params)
    

class TrainerContext:
    """
    Context responsible for orchestrating the training process
    using a given TrainingStrategy.
    """

    def __init__(self, strategy: TrainingStrategy):
        self.strategy = strategy

    def train(self) -> Path:
        """
        Execute the training pipeline:
        - Build data pipeline
        - Infer input size from data
        - Build model via factory
        - Train using PyTorch Lightning
        - Save trained model
        """

        # -----------------------------------------------------------------
        # Retrieve training parameters
        # -----------------------------------------------------------------
        params = self.strategy.get_training_params()

        # -----------------------------------------------------------------
        # Logger
        # -----------------------------------------------------------------
        print(params.get("experiment_name", "default_experiment"))
        mlflow_logger = MLFlowLogger(
            experiment_name=params.get("experiment_name", "default_experiment"),
            run_name=self.strategy.name,
        )

        experiment = mlflow_logger.experiment
        run_id = mlflow_logger.run_id

        # -----------------------------------------------------------------
        # Log EXPERIMENT IDENTITY (TAGS)
        # -----------------------------------------------------------------
        experiment.set_tag(run_id, "training_strategy", self.strategy.name)
        experiment.set_tag(
            run_id,
            "experiment_type",
            "ray_search" if params.get("is_ray_run", False) else "standard_train",
        )

        # -----------------------------------------------------------------
        # Log TRAINING PARAMS (PARAMS)
        # -----------------------------------------------------------------
        experiment.log_param(run_id, "learning_rate", params["learning_rate"])
        experiment.log_param(run_id, "batch_size", params["batch_size"])
        experiment.log_param(run_id, "num_epochs", params["num_epochs"])
        experiment.log_param(run_id, "optimizer", "Adam")
        experiment.log_param(run_id, "loss_function", "MSELoss")

        # -----------------------------------------------------------------
        # Data pipeline (FIRST)
        # -----------------------------------------------------------------
        pipeline = self.strategy.get_data_pipeline(
            mlflow_logger=mlflow_logger,
            is_ray_run=params.get("is_ray_run", False),
        )

        train_loader, val_loader, _ = pipeline.run(
            tickers=params["tickers"],
            period=params["period"],
            seq_len=params["seq_len"],
            train_ratio=params["train_ratio"],
            val_ratio=params["val_ratio"],
        )

        # -----------------------------------------------------------------
        # Infer input size from pipeline
        # -----------------------------------------------------------------
        if not hasattr(pipeline, "num_features"):
            raise RuntimeError(
                "DataPipeline must expose 'num_features' "
                "to build a compatible model."
            )

        input_size = pipeline.num_features
        experiment.log_param(run_id, "input_size", input_size)

        # -----------------------------------------------------------------
        # Model creation (AFTER data)
        # -----------------------------------------------------------------
        factory = self.strategy.get_model_factory(
            input_size=input_size
        )

        model = factory.create()

        # -----------------------------------------------------------------
        # Log MODEL ARCHITECTURE (PARAMS)
        # -----------------------------------------------------------------
        experiment.log_param(run_id, "model_type", model.__class__.__name__)

        if hasattr(factory, "params"):
            model_params = factory.params
            experiment.log_param(run_id, "hidden_size", model_params.hidden_size)
            experiment.log_param(run_id, "num_layers", model_params.num_layers)
            experiment.log_param(run_id, "dropout", model_params.dropout)
            experiment.log_param(run_id, "output_size", model_params.output_size)

        experiment.log_param(run_id, "layer_config", params["layer_config"])

        # -----------------------------------------------------------------
        # Lightning module
        # -----------------------------------------------------------------
        lightning_module = LSTMLightningModule(
            model=model,
            lr=params["learning_rate"],
        )

        # -----------------------------------------------------------------
        # Trainer
        # -----------------------------------------------------------------
        trainer = pl.Trainer(
            max_epochs=params["num_epochs"],
            logger=mlflow_logger,
            enable_checkpointing=True,
            default_root_dir=str(Path(".").resolve()),
        )

        # -----------------------------------------------------------------
        # Training
        # -----------------------------------------------------------------
        trainer.fit(
            lightning_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )

        # -----------------------------------------------------------------
        # Save final model (ARTIFACT)
        # -----------------------------------------------------------------
        if params.get("is_ray_run", False):
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = Path(tmpdir) / f"{self.strategy.name}.pt"
                torch.save(model.state_dict(), model_path)

                experiment.log_artifact(run_id, str(model_path))

        else:
            model_dir = Path("artifacts/models")
            model_dir.mkdir(exist_ok=True)

            model_path = model_dir / f"model_final.pt"
            torch.save(model.state_dict(), model_path)

            experiment.log_artifact(run_id, str(model_path))

            return model_path