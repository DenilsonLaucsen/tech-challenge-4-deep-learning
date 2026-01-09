from src.training.model import TrainerContext, NoProcessingSingleStrategy, TrainingParams
from src.models.lstm_params import LSTMParams
from src.data.scaler import TimeSeriesScaler


def main():
    lstm_params = LSTMParams(
        input_size=4,
        hidden_size=64,
        num_layers=1,
        output_size=1,
        dropout=0.0,
    )

    training_params = TrainingParams(
        tickers=["AAPL"],
        period="1y",
        seq_len=20,
        num_epochs=5,
        learning_rate=1e-3,
        batch_size=16,
        layer_config=["LSTM", "Linear"],
        lstm_params=lstm_params,
        shuffle=True,
        scaler=TimeSeriesScaler(),
        scaler_name="scaler_smoke_train",
        enable_mlflow=True,
        train_ratio=0.7,
        val_ratio=0.15,
        is_ray_run=False,
        experiment_name='smoke_test_experiment',
    )

    strategy = NoProcessingSingleStrategy(training_params)
    trainer = TrainerContext(strategy)

    model_path = trainer.train()

    print("Smoke training OK ✅")
    print("Model saved at:", model_path)


if __name__ == "__main__":
    main()
