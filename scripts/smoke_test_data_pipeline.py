import mlflow

from src.data.data import DataPipeline, NoProcessingSingle
from src.data.scaler import TimeSeriesScaler


def main():
    mlflow.set_experiment("data-pipeline-smoke-test")

    with mlflow.start_run(run_name="smoke_test_pipeline"):
        strategy = NoProcessingSingle()

        pipeline = DataPipeline(
            strategy=strategy,
            batch_size=16,
            shuffle=True,
            scaler=TimeSeriesScaler(),
            scaler_name="scaler_smoke_test_data_pipeline",
            enable_mlflow=True,
            mlflow_logger=mlflow.get_logger(),
        )

        train_loader, val_loader, test_loader = pipeline.run(
            tickers=["AAPL"],
            period="1y",
            seq_len=20,
            train_ratio=0.7,
            val_ratio=0.15,
        )

        x_batch, y_batch = next(iter(train_loader))
        print("Smoke test OK")
        print("X batch shape:", x_batch.shape)
        print("Y batch shape:", y_batch.shape)


if __name__ == "__main__":
    main()
