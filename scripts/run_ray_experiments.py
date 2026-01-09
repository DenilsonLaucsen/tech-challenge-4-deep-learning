import yaml
import itertools

import ray

from src.training.model import (
    TrainerContext,
    TrainingParams,
    NoProcessingSingleStrategy,
    NoProcessingMultipleStrategy,
    RangeSingleStrategy,
    RangeMultipleStrategy,
)
from src.models.lstm_params import LSTMParams
from src.data.scaler import TimeSeriesScaler


# ---------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------
STRATEGY_REGISTRY = {
    "NoProcessingSingleStrategy": NoProcessingSingleStrategy,
    "NoProcessingMultipleStrategy": NoProcessingMultipleStrategy,
    "RangeSingleStrategy": RangeSingleStrategy,
    "RangeMultipleStrategy": RangeMultipleStrategy,
}


# ---------------------------------------------------------------------
# Ray train job (1 experimento válido = 1 execução completa)
# ---------------------------------------------------------------------
@ray.remote
def run_training_experiment(
    strategy_name: str,
    fixed_params: dict,
    search_params: dict,
    experiment_name: str
):
    """
    Executes a single valid training experiment.
    Each Ray job runs a full training pipeline via TrainerContext.
    """

    # -----------------------------------------------------------------
    # Build LSTM params (architecture only, input_size inferred later)
    # -----------------------------------------------------------------
    lstm_params = LSTMParams(
        hidden_size=search_params["hidden_size"],
        num_layers=search_params["num_layers"],
        output_size=1,
        dropout=0.0,
    )

    # -----------------------------------------------------------------
    # Build training params
    # -----------------------------------------------------------------
    training_params = TrainingParams(
        tickers=fixed_params["tickers"],
        period=fixed_params["period"],
        seq_len=search_params["seq_len"],
        num_epochs=fixed_params["num_epochs"],
        learning_rate=search_params["learning_rate"],
        batch_size=search_params["batch_size"],
        layer_config=["LSTM", "Linear"],
        lstm_params=lstm_params,
        shuffle=True,
        scaler=TimeSeriesScaler(),
        scaler_name=f"scaler_{strategy_name}",
        enable_mlflow=True,
        train_ratio=fixed_params["train_ratio"],
        val_ratio=fixed_params["val_ratio"],
        is_ray_run=True, 
        experiment_name=experiment_name,
    )

    # -----------------------------------------------------------------
    # Instantiate strategy
    # -----------------------------------------------------------------
    strategy_cls = STRATEGY_REGISTRY[strategy_name]
    strategy = strategy_cls(training_params)

    # -----------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------
    trainer = TrainerContext(strategy)
    model_path = trainer.train()

    return {
        "strategy": strategy_name,
        "params": search_params,
        "model_path": str(model_path),
    }


# ---------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------
def main():
    # -----------------------------------------------------------------
    # Load config
    # -----------------------------------------------------------------
    with open("configs/ray_experiments.yaml", "r") as f:
        config = yaml.safe_load(f)

    # -----------------------------------------------------------------
    # Ray init
    # -----------------------------------------------------------------
    ray.init(
        num_cpus=config["ray"]["num_cpus"],
        num_gpus=config["ray"]["num_gpus"],
        ignore_reinit_error=True,
    )

    print("🚀 Ray initialized")

    strategies = config["strategies"]
    search_space = config["search_space"]
    fixed_params = config["fixed_params"]
    experiment_name = config["experiment"]["name"]

    # -----------------------------------------------------------------
    # Build cartesian product of search space
    # -----------------------------------------------------------------
    search_keys = list(search_space.keys())
    search_values = list(search_space.values())

    param_combinations = [
        dict(zip(search_keys, values))
        for values in itertools.product(*search_values)
    ]

    print(f"🔬 Total parameter combinations: {len(param_combinations)}")
    print(f"🧠 Total strategies (declared): {len(strategies)}")

    # -----------------------------------------------------------------
    # Filter VALID experiments
    # -----------------------------------------------------------------
    valid_experiments = []

    for strategy_name in strategies:
        strategy_cls = STRATEGY_REGISTRY[strategy_name]

        for search_params in param_combinations:
            # Temporary params ONLY for compatibility validation
            tmp_params = TrainingParams(
                tickers=fixed_params["tickers"],
                period=fixed_params["period"],
                seq_len=search_params["seq_len"],
                num_epochs=fixed_params["num_epochs"],
                learning_rate=search_params["learning_rate"],
                batch_size=search_params["batch_size"],
                layer_config=["LSTM", "Linear"],
                lstm_params=None,
                shuffle=True,
                scaler=None,
                scaler_name=None,
                enable_mlflow=False,
                train_ratio=fixed_params["train_ratio"],
                val_ratio=fixed_params["val_ratio"],
                is_ray_run=True,
            )

            if strategy_cls.is_compatible(tmp_params):
                valid_experiments.append((strategy_name, search_params))

    print(f"✅ Valid experiments: {len(valid_experiments)}")

    # -----------------------------------------------------------------
    # Launch Ray jobs (ONLY valid experiments)
    # -----------------------------------------------------------------
    futures = [
        run_training_experiment.remote(
            strategy_name=strategy_name,
            fixed_params=fixed_params,
            search_params=search_params,
            experiment_name=experiment_name
        )
        for strategy_name, search_params in valid_experiments
    ]

    # -----------------------------------------------------------------
    # Collect results
    # -----------------------------------------------------------------
    results = ray.get(futures)

    print("🎉 All Ray experiments completed successfully")
    for result in results:
        print(result)

    ray.shutdown()


if __name__ == "__main__":
    main()
