"""
Training utilities and functions for track prediction models.

This module contains training logic for different approaches including
standard training, ensemble training, hyperparameter tuning, and cross-validation.
"""

import json
import time
from typing import Any, List, cast

import keras
import keras_tuner as kt
import numpy as np
import tensorflow as tf
from keras import Model
from tqdm import tqdm

from imt_ml.dataset import load_tfrecord_dataset
from imt_ml.models import (
    _create_embedding_layers,
    _create_input_layers,
    _create_time_features,
    build_model_from_config,
    build_tunable_model,
    create_simple_model,
)


def _maybe_import_ray_train() -> tuple[Any, Any, Any] | None:
    """Lazily import ray.train TensorFlow trainer and configs.

    Returns (TensorflowTrainer, ScalingConfig, RunConfig) if available, else None.
    """
    try:  # Import lazily to avoid hard dependency at module import time
        from ray.air.config import RunConfig, ScalingConfig  # type: ignore
        from ray.train.tensorflow import TensorflowTrainer  # type: ignore

        return TensorflowTrainer, ScalingConfig, RunConfig
    except Exception:
        return None


def _ray_tf_train(
    *,
    data_dir: str,
    epochs: int,
    batch_size: int,
    model_save_path: str,
    model_kind: str,
    learning_rate: float = 0.001,
    model_config: dict[str, Any] | None = None,
    use_class_weights: bool = True,
    num_workers: int = 1,
    use_gpu: bool = False,
) -> dict[str, Any]:
    """Run training with ray.train (TensorflowTrainer) and return result info.

    Saves a Keras model inside the Ray checkpoint and then writes
    `<model_save_path>_final.keras` in the driver after training completes.
    """
    maybe = _maybe_import_ray_train()
    if maybe is None:
        raise RuntimeError(
            "ray[train] is required for Ray-based training. Ensure it is installed."
        )

    TensorflowTrainer, ScalingConfig, RunConfig = maybe

    # Define per-worker train loop
    def train_loop_per_worker(config: dict[str, Any]) -> None:
        # Best-effort device setup to avoid GPU allocator issues
        try:
            import os as _os

            from ray.train import get_context as _get_context  # type: ignore

            world_rank = _get_context().get_world_rank() if _get_context() else 0
            if not config.get("use_gpu", False):
                _os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
                try:
                    tf.config.set_visible_devices([], "GPU")
                except Exception:
                    pass
            else:
                try:
                    gpus = tf.config.list_physical_devices("GPU")
                    for g in gpus:
                        try:
                            tf.config.experimental.set_memory_growth(g, True)
                        except Exception:
                            pass
                    if world_rank == 0:
                        logical = tf.config.list_logical_devices("GPU")
                        print(
                            f"[Ray Train] GPUs: {len(gpus)} physical, {len(logical)} logical"
                        )
                except Exception:
                    pass
        except Exception:
            pass

        # Load dataset per worker
        train_ds, val_ds, vocab_info = load_tfrecord_dataset(
            config["data_dir"], batch_size=config["batch_size"], augment_data=True
        )

        # Build model
        if config["model_kind"] == "simple":
            model = create_simple_model(vocab_info, use_regularization=True)
        elif config["model_kind"] == "from_config":
            model = build_model_from_config(config["model_config"], vocab_info)
        else:
            raise ValueError(f"Unknown model_kind: {config['model_kind']}")

        optimizer = keras.optimizers.Adam(learning_rate=config["learning_rate"])
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Report metrics back to Ray Train
        class _RayTrainReportCallback(keras.callbacks.Callback):  # type: ignore[misc]
            def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None):
                try:
                    from ray.train import session as _session  # type: ignore

                    logs = logs or {}
                    _session.report(  # type: ignore[attr-defined]
                        {
                            "epoch": int(epoch),
                            "accuracy": float(logs.get("accuracy", 0.0)),
                            "loss": float(logs.get("loss", 0.0)),
                            "val_accuracy": float(logs.get("val_accuracy", 0.0)),
                            "val_loss": float(logs.get("val_loss", 0.0)),
                        }
                    )
                except Exception:
                    pass

        callbacks: list[keras.callbacks.Callback] = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=2, restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
            ),
            _RayTrainReportCallback(),
        ]

        fit_kwargs: dict[str, Any] = {
            "x": train_ds,
            "validation_data": val_ds,
            "epochs": int(config["epochs"]),
            "steps_per_epoch": vocab_info["train_steps_per_epoch"],
            "validation_steps": vocab_info["val_steps_per_epoch"],
            "callbacks": callbacks,
            "verbose": 1 if (config.get("verbose", False)) else 0,
        }
        if config.get("use_class_weights", True) and "class_weights" in vocab_info:
            fit_kwargs["class_weight"] = vocab_info["class_weights"]

        model.fit(**fit_kwargs)

        # Save final model inside checkpoint directory
        import os
        import tempfile

        from ray.air import Checkpoint  # type: ignore
        from ray.train import session as _session  # type: ignore

        ckpt_dir = tempfile.mkdtemp(prefix="ray_tf_model_")
        model_out = os.path.join(ckpt_dir, "final_model.keras")
        model.save(model_out, include_optimizer=False)

        # Final evaluation for completeness
        val_loss, val_acc = model.evaluate(
            val_ds, steps=vocab_info["val_steps_per_epoch"], verbose=0
        )
        _session.report(  # type: ignore[attr-defined]
            {"final_val_accuracy": float(val_acc), "final_val_loss": float(val_loss)},
            checkpoint=Checkpoint.from_directory(ckpt_dir),
        )

    scaling_config = ScalingConfig(num_workers=num_workers, use_gpu=use_gpu)

    # Put results under project output directory for easier discovery by users
    run_config = RunConfig(name=model_save_path.rsplit("/", 1)[-1])

    trainer = TensorflowTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={
            "data_dir": data_dir,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "model_kind": model_kind,
            "model_config": model_config or {},
            "use_class_weights": use_class_weights,
            "use_gpu": use_gpu,
            "verbose": True,
        },
        scaling_config=scaling_config,
        run_config=run_config,
    )

    result = trainer.fit()

    # Persist the checkpointed Keras model to the expected path
    model_path = f"{model_save_path}_final.keras"
    if result.checkpoint is not None:
        import os
        import shutil
        import tempfile

        tmpdir = tempfile.mkdtemp(prefix="ray_tf_ckpt_")
        local_dir = result.checkpoint.to_directory(tmpdir)
        src_model = os.path.join(local_dir, "final_model.keras")
        if os.path.exists(src_model):
            shutil.move(src_model, model_path)

    metrics = dict(result.metrics)
    metrics["model_path"] = model_path
    return metrics


class _ModelCheckpointNoOptimizer(keras.callbacks.Callback):  # type: ignore[misc]
    """Checkpoint that saves the full model without optimizer state.

    Mirrors the common `ModelCheckpoint(save_best_only=True)` behavior but
    calls `model.save(..., include_optimizer=False)` to avoid optimizer
    variable mismatch warnings on load.
    """

    def __init__(
        self,
        filepath: str,
        monitor: str = "val_accuracy",
        mode: str = "max",
        save_best_only: bool = True,
        verbose: int = 1,
    ) -> None:
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        if mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'")
        self.best = -np.inf if mode == "max" else np.inf

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return
        improved = current > self.best if self.mode == "max" else current < self.best
        if self.save_best_only and not improved:
            return
        self.best = current
        if self.verbose:
            print(f"Epoch {epoch + 1}: saving model to {self.filepath} (no optimizer)")
        # Save the full model without optimizer state
        self.model.save(self.filepath, include_optimizer=False)


def tune_hyperparameters(
    data_dir: str,
    project_name: str = "track_prediction_tuning",
    max_epochs: int = 50,
    factor: int = 3,
    hyperband_iterations: int = 1,
    batch_size: int = 32,
    distributed: bool = False,
    executions_per_trial: int = 2,
    directory: str | None = None,
) -> kt.Tuner:
    """Tune hyperparameters using Keras Tuner Hyperband algorithm."""

    # Load dataset
    print(f"Loading dataset from {data_dir}...")
    train_ds, val_ds, vocab_info = load_tfrecord_dataset(
        data_dir, batch_size=batch_size
    )

    print("Vocabulary sizes:")
    print(f"  Stations: {vocab_info['num_stations']}")
    print(f"  Routes: {vocab_info['num_routes']}")
    print(f"  Tracks: {vocab_info['num_tracks']}")

    if distributed:
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        with strategy.scope():
            tuner = kt.Hyperband(
                hypermodel=lambda hp: build_tunable_model(hp, vocab_info),
                objective="val_accuracy",
                max_epochs=max_epochs,
                factor=factor,
                hyperband_iterations=hyperband_iterations,
                executions_per_trial=executions_per_trial,
                project_name=project_name,
                overwrite=True,
                directory=directory,
                distribution_strategy=strategy,
            )
    else:
        tuner = kt.Hyperband(
            hypermodel=lambda hp: build_tunable_model(hp, vocab_info),
            objective="val_accuracy",
            max_epochs=max_epochs,
            factor=factor,
            hyperband_iterations=hyperband_iterations,
            executions_per_trial=executions_per_trial,
            project_name=project_name,
            overwrite=True,
            directory=directory,
        )

    print(
        f"Starting Hyperband hyperparameter tuning with max_epochs={max_epochs}, factor={factor}..."
    )
    print("Search space:")
    tuner.search_space_summary()

    # Setup callbacks for tuning
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
        ),
        # During hyperparameter tuning, keep optimizer state in checkpoints
        keras.callbacks.ModelCheckpoint(
            filepath=f"{project_name}_best.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # Search for best hyperparameters
    print("Starting hyperparameter search...")
    tuner.search(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=vocab_info["train_steps_per_epoch"],
        validation_steps=vocab_info["val_steps_per_epoch"],
        callbacks=callbacks,
        verbose=1,
    )

    # Print results
    print("\nTuning completed!")
    print("Best hyperparameters:")
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    for param in best_hps.space:
        print(f"  {param.name}: {best_hps.get(param.name)}")

    return tuner


def tune_hyperparameters_ray(
    data_dir: str,
    project_name: str = "track_prediction_tuning",
    max_epochs: int = 50,
    batch_size: int = 32,
    num_samples: int = 20,
    asha_grace_period: int = 5,
    asha_reduction_factor: int = 3,
    directory: str | None = None,
    gpus_per_trial: float = 0.0,
    cpus_per_trial: float | None = None,
    *,
    use_bohb: bool = False,
    bohb_max_concurrent: int | None = None,
    use_pbt: bool = False,
    pbt_perturbation_interval: int = 3,
) -> dict[str, Any]:
    """Tune hyperparameters using Ray Tune.

    Defaults to ASHA scheduler. If `use_bohb` is True, uses BOHB
    (HyperBandForBOHB + TuneBOHB search algorithm).

    Returns the best config dict found by Ray.
    """
    # Import Ray lazily to avoid hard dependency at import time
    try:
        import os as _os

        import ray  # type: ignore
        from ray import tune  # type: ignore
        from ray.tune.schedulers import (
            ASHAScheduler,
            HyperBandForBOHB,
            PopulationBasedTraining,
        )  # type: ignore

        if use_bohb:
            # Imported lazily only if requested
            from ray.tune.search import ConcurrencyLimiter  # type: ignore
            from ray.tune.search.bohb import TuneBOHB  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Ray Tune is required for tune_hyperparameters_ray. Install ray[tune]."
        ) from e

    # Ensure Ray uses the active virtualenv and doesn't try to recreate envs with uv
    _os.environ.setdefault("UV_ACTIVE", "1")
    if not ray.is_initialized():  # type: ignore[attr-defined]
        ray.init(runtime_env={"env_vars": {"UV_ACTIVE": "1"}})  # type: ignore[call-arg]

    def trainable(config: dict[str, Any]):
        # Avoid TensorFlow GPU initialization issues in Ray workers when not using GPUs.
        # Must be done before any TensorFlow ops create tensors or access devices.
        try:  # defensive: never let TF device config crash the worker
            import os as _os

            if float(gpus_per_trial or 0.0) == 0.0:
                # Hide GPUs from TF entirely for CPU-only trials
                _os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
                try:
                    tf.config.set_visible_devices([], "GPU")
                except Exception:
                    # set_visible_devices may raise if called after devices are initialized
                    pass
            else:
                # When using GPUs, enable memory growth to reduce OOM / allocator issues
                try:
                    _gpus = tf.config.list_physical_devices("GPU")
                    for _gpu in _gpus:
                        try:
                            tf.config.experimental.set_memory_growth(_gpu, True)
                        except Exception:
                            pass
                    # Log visible GPUs for transparency
                    _logical = tf.config.list_logical_devices("GPU")
                    print(
                        f"[Ray Trial] Visible GPUs: {len(_gpus)} physical, {len(_logical)} logical"
                    )
                    if _gpus:
                        names = [getattr(g, "name", "GPU") for g in _gpus]
                        print(f"[Ray Trial] GPU devices: {names}")
                except Exception:
                    pass
        except Exception:
            pass

        # Load dataset per trial to avoid cross-trial state
        train_ds, val_ds, vocab_info = load_tfrecord_dataset(
            data_dir, batch_size=batch_size
        )
        # Build and train model for this config
        model = build_model_from_config(config, vocab_info)

        # Report metrics to Ray each epoch
        class _TuneReportCallback(keras.callbacks.Callback):  # type: ignore[misc]
            def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None):
                logs = logs or {}
                metrics = {
                    "val_accuracy": float(logs.get("val_accuracy", 0.0)),
                    "val_loss": float(logs.get("val_loss", 0.0)),
                    "accuracy": float(logs.get("accuracy", 0.0)),
                    "loss": float(logs.get("loss", 0.0)),
                    "epoch": int(epoch),
                }
                # Report metrics as top-level keys so schedulers can see `val_accuracy`
                # Ray Train/Tune in this environment expects a single metrics dict.
                # Passing kwargs (e.g., val_accuracy=...) raises TypeError.
                tune.report(metrics)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            ),
            _TuneReportCallback(),
        ]

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=max_epochs,
            steps_per_epoch=vocab_info["train_steps_per_epoch"],
            validation_steps=vocab_info["val_steps_per_epoch"],
            callbacks=callbacks,
            verbose=0,
        )

    # Define search space similar to Keras Tuner configuration
    max_layers = 10
    config_space: dict[str, Any] = {
        "station_embedding_dim": tune.choice([8, 16, 24, 32, 40, 48, 56, 64]),
        "route_embedding_dim": tune.choice([4, 8, 12, 16, 24, 32, 40, 48, 56, 64]),
        "direction_embedding_dim": tune.choice([2, 4, 6, 8, 12, 16, 24, 32]),
        "num_layers": tune.randint(2, max_layers + 1),
        "learning_rate": tune.loguniform(1e-4, 1e-2),
    }
    for i in range(max_layers):
        config_space[f"units_{i}"] = tune.randint(8, 257)
        config_space[f"dropout_{i}"] = tune.uniform(0.01, 0.6)

    scheduler: Any
    search_alg: Any | None
    if use_bohb:
        # BOHB scheduler. Note: grace_period is not used by BOHB.
        scheduler = HyperBandForBOHB(
            time_attr="training_iteration",
            metric="val_accuracy",
            mode="max",
            max_t=max_epochs,
            reduction_factor=asha_reduction_factor,
        )
        # Use BOHB search algorithm as well
        bohb_search = TuneBOHB()
        if bohb_max_concurrent and bohb_max_concurrent > 0:
            search_alg = ConcurrencyLimiter(
                bohb_search, max_concurrent=bohb_max_concurrent
            )
        else:
            search_alg = bohb_search
    elif use_pbt:
        # Population Based Training schedule
        scheduler = PopulationBasedTraining(
            time_attr="training_iteration",
            metric="val_accuracy",
            mode="max",
            perturbation_interval=pbt_perturbation_interval,
            hyperparam_mutations={
                # Explore learning rates
                "learning_rate": [1e-4, 5e-4, 1e-3, 5e-3],
                # Explore embedding sizes
                "station_embedding_dim": [8, 16, 24, 32, 40, 48, 56, 64],
                "route_embedding_dim": [4, 8, 12, 16, 24, 32, 40, 48, 56, 64],
                "direction_embedding_dim": [2, 4, 6, 8, 12, 16, 24, 32],
                # Explore depth
                "num_layers": list(range(2, max_layers + 1)),
            },
        )
        search_alg = None
    else:
        # Default scheduler: ASHA
        scheduler = ASHAScheduler(
            time_attr="training_iteration",
            metric="val_accuracy",
            mode="max",
            grace_period=asha_grace_period,
            reduction_factor=asha_reduction_factor,
            max_t=max_epochs,
        )
        search_alg = None

    trainable_with_resources = tune.with_resources(
        trainable, resources={"cpu": (cpus_per_trial or 1), "gpu": gpus_per_trial}
    )

    run_kwargs: dict[str, Any] = {
        "name": project_name,
        "scheduler": scheduler,
        "num_samples": num_samples,
        "local_dir": directory,  # may be None, Ray defaults to ~/ray_results
        "config": config_space,
        "verbose": 1,
    }
    if search_alg is not None:
        run_kwargs["search_alg"] = search_alg

    analysis = tune.run(
        trainable_with_resources,
        **run_kwargs,
    )

    # Explicitly provide metric/mode when fetching best config to avoid Tune API warnings
    best_config = cast(
        dict[str, Any], analysis.get_best_config(metric="val_accuracy", mode="max")
    )
    # Persist best config as JSON next to Ray results for convenience
    try:
        import json
        import os

        out_dir = analysis.get_best_logdir("val_accuracy", mode="max")
        if out_dir:
            with open(os.path.join(out_dir, "best_config.json"), "w") as f:
                json.dump(best_config, f, indent=2)
    except Exception:
        pass

    return best_config


def train_best_model_from_config(
    best_config: dict[str, Any],
    data_dir: str,
    epochs: int = 50,
    batch_size: int = 32,
    model_save_path: str = "track_prediction_model_tuned",
    tuning_time: float = 0.0,
    generate_report_func=None,
    *,
    use_ray_train: bool = False,
    num_workers: int = 1,
    use_gpu: bool = False,
    tuning_algorithm: str | None = None,
) -> keras.Model:
    """Train a model using a Ray Tune best config and save artifacts."""
    start_time = time.time()

    # Load dataset (driver) for metadata and evaluation below
    train_ds, val_ds, vocab_info = load_tfrecord_dataset(
        data_dir, batch_size=batch_size
    )

    print("Training best model with Ray-derived hyperparameters:")
    for k, v in best_config.items():
        print(f"  {k}: {v}")

    history_like: dict[str, list[float]] | None = None
    if use_ray_train:
        # Train via Ray Train using the config
        _ = _ray_tf_train(
            data_dir=data_dir,
            epochs=epochs,
            batch_size=batch_size,
            model_save_path=model_save_path,
            model_kind="from_config",
            model_config=best_config,
            learning_rate=float(best_config.get("learning_rate", 1e-3)),
            use_class_weights=True,
            num_workers=num_workers,
            use_gpu=use_gpu,
        )
        # No history timeline from Ray path; will compute metrics below
    else:
        # Local training path
        model = build_model_from_config(best_config, vocab_info)
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=2, restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
            ),
            _ModelCheckpointNoOptimizer(
                filepath=f"{model_save_path}_best.keras",
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
        ]

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            steps_per_epoch=vocab_info["train_steps_per_epoch"],
            validation_steps=vocab_info["val_steps_per_epoch"],
            callbacks=callbacks,
            verbose=1,
        )

        model.save(f"{model_save_path}_final.keras", include_optimizer=False)
        print(f"Model saved to {model_save_path}_final.keras (no optimizer state)")
        history_like = {k: list(v) for k, v in history.history.items()}

    # Calculate total time
    final_training_time = time.time() - start_time
    total_time = tuning_time + final_training_time

    # Generate tuning report if function provided
    if generate_report_func:
        training_params = {
            "epochs": epochs,
            "batch_size": batch_size,
            "dataset_size": vocab_info["metadata"]["total_records"],
            "tuning_algorithm": tuning_algorithm or "Ray Tune ASHA",
        }

        # Save hyperparameters and vocabulary info for traceability
        save_path = f"{model_save_path}_config.json"
        config = {
            "hyperparameters": best_config,
            "vocabulary": {
                "station_vocab": vocab_info["station_vocab"],
                "route_vocab": vocab_info["route_vocab"],
                "track_vocab": vocab_info["track_vocab"],
                "num_stations": vocab_info["num_stations"],
                "num_routes": vocab_info["num_routes"],
                "num_tracks": vocab_info["num_tracks"],
            },
        }
        with open(save_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved to {save_path}")

        # Load the saved model and evaluate for consistent metrics
        loaded = keras.models.load_model(
            f"{model_save_path}_final.keras", compile=False
        )
        loaded.compile(
            optimizer=keras.optimizers.Adam(),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        val_loss, val_accuracy = loaded.evaluate(
            val_ds, steps=vocab_info["val_steps_per_epoch"], verbose=0
        )

        final_metrics = {
            "validation_loss": float(val_loss),
            "validation_accuracy": float(val_accuracy),
            "total_epochs_trained": (
                len(history_like["loss"]) if history_like else epochs
            ),
            "best_validation_accuracy": (
                max(history_like.get("val_accuracy", [0.0]))
                if history_like
                else float(val_accuracy)
            ),
            "best_validation_loss": (
                min(history_like.get("val_loss", [float(val_loss)]))
                if history_like
                else float(val_loss)
            ),
        }

        additional_info = {
            "final_training_time": final_training_time,
            "total_time": total_time,
            "model_parameters": loaded.count_params(),
            "optimization_objective": "val_accuracy",
            "best_hyperparameters": best_config,
        }

        generate_report_func(
            "tune",
            model_save_path,
            vocab_info,
            training_params,
            final_metrics,
            total_time,
            additional_info,
        )

    return keras.models.load_model(f"{model_save_path}_final.keras", compile=False)


def train_best_model(
    tuner: kt.Tuner,
    data_dir: str,
    epochs: int = 50,
    batch_size: int = 32,
    model_save_path: str = "track_prediction_model_tuned",
    tuning_time: float = 0.0,
    generate_report_func=None,
) -> keras.Model:
    """Train the best model found by hyperparameter tuning."""
    start_time = time.time()

    # Load dataset
    print(f"Loading dataset from {data_dir}...")
    train_ds, val_ds, vocab_info = load_tfrecord_dataset(
        data_dir, batch_size=batch_size
    )

    # Get best hyperparameters and build model
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = build_tunable_model(best_hps, vocab_info)

    print("Training best model with hyperparameters:")
    best_hp_dict = {}
    for param in best_hps.space:
        value = best_hps.get(param.name)
        print(f"  {param.name}: {value}")
        best_hp_dict[param.name] = value

    print(model.summary())

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
        ),
        _ModelCheckpointNoOptimizer(
            filepath=f"{model_save_path}_best.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print(f"Training model for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=vocab_info["train_steps_per_epoch"],
        validation_steps=vocab_info["val_steps_per_epoch"],
        callbacks=callbacks,
        verbose=1,
    )

    model.save(f"{model_save_path}_final.keras", include_optimizer=False)
    print(f"Model saved to {model_save_path}_final.keras (no optimizer state)")

    # Save hyperparameters and vocabulary info
    save_path = f"{model_save_path}_config.json"
    config = {
        "hyperparameters": best_hp_dict,
        "vocabulary": {
            "station_vocab": vocab_info["station_vocab"],
            "route_vocab": vocab_info["route_vocab"],
            "track_vocab": vocab_info["track_vocab"],
            "num_stations": vocab_info["num_stations"],
            "num_routes": vocab_info["num_routes"],
            "num_tracks": vocab_info["num_tracks"],
        },
    }
    with open(save_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {save_path}")

    # Evaluate model
    print("\nFinal evaluation:")
    val_loss, val_accuracy = model.evaluate(
        val_ds, steps=vocab_info["val_steps_per_epoch"], verbose=1
    )
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Calculate total time
    final_training_time = time.time() - start_time
    total_time = tuning_time + final_training_time

    # Generate tuning report if function provided
    if generate_report_func:
        training_params = {
            "epochs": epochs,
            "batch_size": batch_size,
            "dataset_size": vocab_info["metadata"]["total_records"],
            "tuning_algorithm": "Hyperband",
        }
        training_params.update(best_hp_dict)  # Add best hyperparameters

        final_metrics = {
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
            "total_epochs_trained": len(history.history["loss"]),
            "best_validation_accuracy": max(history.history["val_accuracy"]),
            "best_validation_loss": min(history.history["val_loss"]),
        }

        additional_info = {
            "hyperparameter_search_time": tuning_time,
            "final_training_time": final_training_time,
            "total_time": total_time,
            "model_parameters": model.count_params(),
            "optimization_objective": "val_accuracy",
            "search_algorithm": "Hyperband with early stopping",
            "best_hyperparameters": best_hp_dict,
        }

        generate_report_func(
            "tune",
            model_save_path,
            vocab_info,
            training_params,
            final_metrics,
            total_time,
            additional_info,
        )

    return model


def _create_optimized_callbacks(
    model_save_path: str,
    dataset_size: int,
    monitor_patience: int | None = None,
    epochs: int = 50,
    *,
    min_delta: float = 1e-4,
    reduce_lr_patience: int | None = None,
    reduce_lr_factor: float = 0.5,
    min_lr: float = 1e-7,
    use_scheduler: bool = True,
    scheduler_tmax: int = 50,
    use_early_stopping: bool = True,
) -> list[keras.callbacks.Callback]:
    """Create optimized callbacks for small datasets."""
    # Adjust patience based on dataset size
    if monitor_patience is None:
        monitor_patience = (
            1 + round(epochs * 0.04)
            if dataset_size < 5000
            else 1 + round(epochs * 0.05)
        )

    # Ensure we have valid patience value before proceeding
    actual_patience = monitor_patience

    callbacks: list[keras.callbacks.Callback] = []

    if use_early_stopping:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=actual_patience,
                restore_best_weights=True,
                verbose=1,
                min_delta=min_delta,
            )
        )

    # Set default patience for ReduceLROnPlateau if not provided
    rlrop_patience = (
        round((2 + actual_patience) / 2)
        if reduce_lr_patience is None
        else reduce_lr_patience
    )
    callbacks.append(
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=reduce_lr_factor,
            patience=rlrop_patience,
            min_lr=min_lr,
            verbose=1,
            min_delta=min_delta,
        )
    )

    callbacks.append(
        _ModelCheckpointNoOptimizer(
            filepath=f"{model_save_path}_best.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        )
    )

    if use_scheduler:
        # Cosine annealing scheduler with configurable period; scales from current lr
        callbacks.append(
            keras.callbacks.LearningRateScheduler(
                lambda epoch, lr: lr
                * (0.5 * (1 + np.cos(np.pi * epoch / scheduler_tmax))),
                verbose=1,
            )
        )

    return callbacks


def train_model(
    data_dir: str,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    model_save_path: str = "track_prediction_model",
    use_class_weights: bool = True,
    generate_report_func=None,
) -> Any:
    """Train the track prediction model with optimizations for small datasets."""
    start_time = time.time()

    # Load dataset with augmentation
    print(f"Loading dataset from {data_dir}...")
    train_ds, val_ds, vocab_info = load_tfrecord_dataset(
        data_dir, batch_size=batch_size, augment_data=True
    )

    print("Vocabulary sizes:")
    print(f"  Stations: {vocab_info['num_stations']}")
    print(f"  Routes: {vocab_info['num_routes']}")
    print(f"  Tracks: {vocab_info['num_tracks']}")

    # Decide on local vs Ray Train path
    use_ray_env = False
    num_workers = 1
    use_gpu = False
    # Allow enabling via environment flags without changing signature (non-breaking)
    # IMT_RAY_TRAIN=1 enables Ray Train with defaults; IMT_RAY_WORKERS/IMT_RAY_GPU tweak it.
    try:
        import os as _os

        use_ray_env = _os.environ.get("IMT_RAY_TRAIN", "0") == "1"
        num_workers = int(_os.environ.get("IMT_RAY_WORKERS", "1"))
        use_gpu = _os.environ.get("IMT_RAY_GPU", "0") == "1"
    except Exception:
        pass

    # Common dataset size for reporting
    dataset_size = vocab_info["metadata"]["total_records"]

    if use_ray_env:
        print(
            f"Using ray.train with num_workers={num_workers}, use_gpu={use_gpu} for training"
        )
        _ = _ray_tf_train(
            data_dir=data_dir,
            epochs=epochs,
            batch_size=batch_size,
            model_save_path=model_save_path,
            model_kind="simple",
            learning_rate=learning_rate,
            use_class_weights=use_class_weights,
            num_workers=num_workers,
            use_gpu=use_gpu,
        )
        # Continue to save metadata and evaluate below using the saved model
    else:
        # Create and compile model with regularization
        print("Creating model with regularization...")
        model = create_simple_model(vocab_info, use_regularization=True)

        # Use adaptive learning rate based on dataset size
        adaptive_lr = learning_rate * (
            0.5 if vocab_info["metadata"]["total_records"] < 5000 else 1.0
        )
        optimizer = keras.optimizers.Adam(
            learning_rate=adaptive_lr,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,  # Smaller epsilon for better precision
        )

        # Compile with class weights if requested
        compile_kwargs = {
            "optimizer": optimizer,
            "loss": "sparse_categorical_crossentropy",
            "metrics": ["accuracy", "top_k_categorical_accuracy"],
        }

        model.compile(**compile_kwargs)

        print(model.summary())

        # Setup optimized callbacks
        dataset_size = vocab_info["metadata"]["total_records"]
        callbacks = _create_optimized_callbacks(
            model_save_path, dataset_size, epochs=epochs
        )

        # Train model with class weights for imbalanced data
        print(f"Training model for {epochs} epochs...")
        fit_kwargs = {
            "x": train_ds,
            "validation_data": val_ds,
            "epochs": epochs,
            "steps_per_epoch": vocab_info["train_steps_per_epoch"],
            "validation_steps": vocab_info["val_steps_per_epoch"],
            "callbacks": callbacks,
            "verbose": 1,
        }

        if use_class_weights and "class_weights" in vocab_info:
            fit_kwargs["class_weight"] = vocab_info["class_weights"]
            print(f"Using class weights for {len(vocab_info['class_weights'])} classes")

        history = model.fit(**fit_kwargs)
        history_like = {k: list(v) for k, v in history.history.items()}

        # Save final model
        model.save(f"{model_save_path}_final.keras", include_optimizer=False)
        print(f"Model saved to {model_save_path}_final.keras (no optimizer state)")

    # Save vocabulary info
    vocab_save_path = f"{model_save_path}_vocab.json"
    with open(vocab_save_path, "w") as f:
        json.dump(
            {
                "station_vocab": vocab_info["station_vocab"],
                "route_vocab": vocab_info["route_vocab"],
                "track_vocab": vocab_info["track_vocab"],
                "num_stations": vocab_info["num_stations"],
                "num_routes": vocab_info["num_routes"],
                "num_tracks": vocab_info["num_tracks"],
                "dataset_size": vocab_info["metadata"]["total_records"],
            },
            f,
            indent=2,
        )
    print(f"Vocabulary saved to {vocab_save_path}")

    # Evaluate saved model
    print("\nFinal evaluation (reloading saved model):")
    loaded_model = keras.models.load_model(
        f"{model_save_path}_final.keras", compile=False
    )
    loaded_model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    val_loss, val_accuracy = loaded_model.evaluate(
        val_ds, steps=vocab_info["val_steps_per_epoch"], verbose=1
    )
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Calculate training time
    training_time = time.time() - start_time

    # Generate training report if function provided
    if generate_report_func:
        training_params = {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "adaptive_learning_rate": (
                adaptive_lr if "adaptive_lr" in locals() else learning_rate
            ),
            "use_class_weights": use_class_weights,
            "dataset_size": dataset_size,
        }

        epochs_trained = (
            len(history_like["loss"]) if "history_like" in locals() else epochs
        )
        final_metrics = {
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
            "total_epochs_trained": epochs_trained,
        }

        # Add best metrics from history
        if "history_like" in locals() and "val_accuracy" in history_like:
            final_metrics["best_validation_accuracy"] = max(
                history_like["val_accuracy"]
            )
            final_metrics["best_validation_loss"] = min(history_like["val_loss"])

        additional_info = {
            "model_parameters": loaded_model.count_params(),
            "regularization": "L1/L2, Dropout, BatchNormalization",
            "data_augmentation": "Applied for small datasets"
            if dataset_size < 10000
            else "Not applied",
        }

        generate_report_func(
            "train",
            model_save_path,
            vocab_info,
            training_params,
            final_metrics,
            training_time,
            additional_info,
        )

    return (
        history
        if "history" in locals()
        else {"validation_loss": val_loss, "validation_accuracy": val_accuracy}
    )


def create_ensemble_model(
    vocab_info: dict[str, Any], num_models: int = 3
) -> list[keras.Model]:
    """Create an ensemble of models with different architectures for better accuracy."""
    models = []

    for i in range(num_models):
        # Create input layers
        inputs = _create_input_layers()

        # Create embeddings with slight variations
        embeddings = _create_embedding_layers(
            inputs, vocab_info, use_regularization=True
        )
        time_features = _create_time_features(inputs)

        # Concatenate features
        all_features = list(embeddings.values()) + time_features
        concat_features = keras.layers.Concatenate()(all_features)
        concat_features = keras.layers.BatchNormalization()(concat_features)

        # Different architectures for diversity
        if i == 0:  # Deeper network
            x = keras.layers.Dense(
                256,
                activation="relu",
                kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
            )(concat_features)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.3)(x)
            x = keras.layers.Dense(
                128,
                activation="relu",
                kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
            )(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.3)(x)
            x = keras.layers.Dense(
                64,
                activation="relu",
                kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
            )(x)
        elif i == 1:  # Wider network
            x = keras.layers.Dense(
                512,
                activation="relu",
                kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
            )(concat_features)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.4)(x)
            x = keras.layers.Dense(
                256,
                activation="relu",
                kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
            )(x)
        else:  # Standard network with different regularization
            x = keras.layers.Dense(
                128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-3)
            )(concat_features)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.5)(x)
            x = keras.layers.Dense(
                64, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-3)
            )(x)

        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.4)(x)

        # Output layer
        outputs = keras.layers.Dense(vocab_info["num_tracks"], activation="softmax")(x)

        # Create model
        model = keras.Model(inputs=list(inputs.values()), outputs=outputs)
        models.append(model)

    return models


def train_ensemble_model(
    data_dir: str,
    num_models: int = 3,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    model_save_path: str = "track_prediction_ensemble",
    generate_report_func=None,
    *,
    early_stop_patience: int | None = None,
    early_stop_min_delta: float = 1e-4,
    reduce_lr_patience: int | None = None,
    reduce_lr_factor: float = 0.5,
    min_lr: float = 1e-7,
    use_scheduler: bool = True,
    scheduler_tmax: int = 50,
    use_early_stopping: bool = True,
) -> list[keras.Model]:
    """Train ensemble of models for improved accuracy on small datasets."""
    start_time = time.time()

    # Load dataset
    print(f"Loading dataset from {data_dir}...")
    train_ds, val_ds, vocab_info = load_tfrecord_dataset(
        data_dir, batch_size=batch_size, augment_data=True
    )

    print("Vocabulary sizes:")
    print(f"  Stations: {vocab_info['num_stations']}")
    print(f"  Routes: {vocab_info['num_routes']}")
    print(f"  Tracks: {vocab_info['num_tracks']}")

    # Create ensemble models
    print(f"Creating ensemble of {num_models} models...")
    models = create_ensemble_model(vocab_info, num_models)

    trained_models: List[Model] = []
    dataset_size = vocab_info["metadata"]["total_records"]
    individual_metrics = []

    for i, model in enumerate(
        tqdm(models, desc="Training ensemble models", unit="model")
    ):
        tqdm.write(f"\nTraining model {i + 1}/{num_models}...")

        # Compile model
        adaptive_lr = learning_rate * (
            0.8 + 0.4 * np.random.random()
        )  # Vary learning rates
        optimizer = keras.optimizers.Adam(learning_rate=adaptive_lr)
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Setup callbacks
        callbacks = _create_optimized_callbacks(
            f"{model_save_path}_model_{i}",
            dataset_size,
            monitor_patience=early_stop_patience,
            epochs=epochs,
            min_delta=early_stop_min_delta,
            reduce_lr_patience=reduce_lr_patience,
            reduce_lr_factor=reduce_lr_factor,
            min_lr=min_lr,
            use_scheduler=use_scheduler,
            scheduler_tmax=scheduler_tmax,
            use_early_stopping=use_early_stopping,
        )

        # Train model
        fit_kwargs = {
            "x": train_ds,
            "validation_data": val_ds,
            "epochs": epochs,
            "steps_per_epoch": vocab_info["train_steps_per_epoch"],
            "validation_steps": vocab_info["val_steps_per_epoch"],
            "callbacks": callbacks,
            "verbose": 1,
        }

        if "class_weights" in vocab_info:
            fit_kwargs["class_weight"] = vocab_info["class_weights"]

        model.fit(**fit_kwargs)

        # Evaluate individual model
        # val_ds repeats indefinitely; bound evaluation by known validation steps
        val_loss, val_accuracy = model.evaluate(
            val_ds, steps=vocab_info["val_steps_per_epoch"], verbose=1
        )
        individual_metrics.append(
            {
                "model_index": i,
                "validation_loss": val_loss,
                "validation_accuracy": val_accuracy,
                "learning_rate": adaptive_lr,
                "parameters": model.count_params(),
            }
        )

        # Save individual model
        model.save(f"{model_save_path}_model_{i}_final.keras", include_optimizer=False)
        trained_models.append(model)
        tqdm.write(f"Model {i + 1} completed - Accuracy: {val_accuracy:.4f}")

    # Evaluate ensemble
    print("\nEvaluating ensemble performance...")
    ensemble_predictions = []

    for model in tqdm(
        trained_models, desc="Evaluating ensemble models", unit="model", leave=False
    ):
        # val_ds is repeated() (infinite); bound predict with steps to avoid hanging
        predictions = model.predict(
            val_ds, steps=vocab_info["val_steps_per_epoch"], verbose=1
        )
        ensemble_predictions.append(predictions)

    # Calculate ensemble metrics
    ensemble_training_time = time.time() - start_time
    avg_val_loss = np.mean([m["validation_loss"] for m in individual_metrics])
    avg_val_accuracy = np.mean([m["validation_accuracy"] for m in individual_metrics])
    best_individual_accuracy = max(
        [m["validation_accuracy"] for m in individual_metrics]
    )

    print(f"Ensemble of {num_models} models created and trained.")
    print(f"Average individual accuracy: {avg_val_accuracy:.4f}")
    print(f"Best individual accuracy: {best_individual_accuracy:.4f}")

    # Generate ensemble training report if function provided
    if generate_report_func:
        training_params = {
            "num_models": num_models,
            "epochs": epochs,
            "batch_size": batch_size,
            "base_learning_rate": learning_rate,
            "dataset_size": dataset_size,
        }

        final_metrics = {
            "average_validation_loss": avg_val_loss,
            "average_validation_accuracy": avg_val_accuracy,
            "best_individual_accuracy": best_individual_accuracy,
            "worst_individual_accuracy": min(
                [m["validation_accuracy"] for m in individual_metrics]
            ),
            "ensemble_std_accuracy": np.std(
                [m["validation_accuracy"] for m in individual_metrics]
            ),
        }

        additional_info = {
            "individual_model_metrics": individual_metrics,
            "ensemble_strategy": "Diverse architectures (deep, wide, standard)",
            "learning_rate_variation": "0.8x to 1.2x base rate with random variation",
            "total_parameters": sum([m["parameters"] for m in individual_metrics]),
        }

        generate_report_func(
            "ensemble",
            model_save_path,
            vocab_info,
            training_params,
            final_metrics,
            ensemble_training_time,
            additional_info,
        )

    return trained_models


def evaluate_with_cross_validation(
    data_dir: str,
    k_folds: int = 5,
    epochs: int = 30,
    batch_size: int = 32,
    model_save_path: str = "cv_evaluation",
    generate_report_func=None,
) -> dict[str, float]:
    """Evaluate model performance using k-fold cross-validation."""
    start_time = time.time()

    print(f"Loading dataset for {k_folds}-fold cross-validation...")
    train_ds, val_ds, vocab_info = load_tfrecord_dataset(
        data_dir, batch_size=batch_size, augment_data=True
    )

    # Collect all data for CV splits
    all_features = []
    all_targets = []

    print("Collecting data for cross-validation...")
    combined_ds = train_ds.concatenate(val_ds)

    for features, target in tqdm(
        combined_ds.take(-1), desc="Collecting CV data", unit="batch", leave=False
    ):
        all_features.append(features)
        all_targets.append(target)

    # total_samples = len(all_features)
    # fold_size = total_samples // k_folds  # For future proper CV implementation

    cv_scores = []
    fold_details = []

    for fold in tqdm(range(k_folds), desc="Cross-validation folds", unit="fold"):
        tqdm.write(f"\nFold {fold + 1}/{k_folds}")

        # Create train/val splits for this fold
        # val_start = fold * fold_size
        # val_end = val_start + fold_size

        # Create model for this fold
        model = create_simple_model(vocab_info, use_regularization=True)

        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Simple training for CV (reduced epochs)
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
            ),
        ]

        # This is a simplified CV implementation
        # In practice, you'd need to properly reconstruct tf.data.Dataset objects
        # For now, we'll use the existing validation approach

        print(f"Training fold {fold + 1}...")
        history = model.fit(
            train_ds.take(vocab_info["train_steps_per_epoch"] * 4 // 5),  # Use subset
            validation_data=val_ds.take(vocab_info["val_steps_per_epoch"] // 2),
            epochs=epochs,
            steps_per_epoch=vocab_info["train_steps_per_epoch"] // 2,
            validation_steps=vocab_info["val_steps_per_epoch"] // 4,
            callbacks=callbacks,
            verbose=1,
        )

        # Get best validation accuracy from history
        best_val_acc = max(history.history["val_accuracy"])
        best_val_loss = min(history.history["val_loss"])
        actual_epochs = len(history.history["loss"])

        fold_details.append(
            {
                "fold": fold + 1,
                "best_validation_accuracy": best_val_acc,
                "best_validation_loss": best_val_loss,
                "epochs_trained": actual_epochs,
                "parameters": model.count_params(),
            }
        )

        cv_scores.append(best_val_acc)
        tqdm.write(f"Fold {fold + 1} accuracy: {best_val_acc:.4f}")

    cv_results = {
        "mean_accuracy": np.mean(cv_scores),
        "std_accuracy": np.std(cv_scores),
        "fold_scores": cv_scores,
    }

    cv_time = time.time() - start_time

    print("\nCross-validation results:")
    print(
        f"Mean accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}"
    )

    # Generate cross-validation report if function provided
    if generate_report_func:
        training_params = {
            "k_folds": k_folds,
            "epochs_per_fold": epochs,
            "batch_size": batch_size,
            "dataset_size": vocab_info["metadata"]["total_records"],
        }

        final_metrics = {
            "mean_cv_accuracy": cv_results["mean_accuracy"],
            "std_cv_accuracy": cv_results["std_accuracy"],
            "best_fold_accuracy": max(cv_scores),
            "worst_fold_accuracy": min(cv_scores),
            "cv_confidence_interval_95": cv_results["std_accuracy"] * 1.96,
        }

        additional_info = {
            "fold_details": fold_details,
            "evaluation_method": "Simplified k-fold (uses existing train/val split per fold)",
            "early_stopping": "Applied with patience=3",
            "total_models_trained": k_folds,
        }

        generate_report_func(
            "cv",
            model_save_path,
            vocab_info,
            training_params,
            final_metrics,
            cv_time,
            additional_info,
        )

    return cv_results
