"""
Training utilities and functions for track prediction models.

This module contains training logic for different approaches including
standard training, ensemble training, hyperparameter tuning, and cross-validation.
"""

import json
import time
from typing import Any

import keras
import keras_tuner as kt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm
from tqdm.keras import TqdmCallback

from imt_ml.dataset import load_tfrecord_dataset
from imt_ml.models import (
    _create_embedding_layers,
    _create_input_layers,
    _create_time_features,
    build_tunable_model,
    create_simple_model,
)


def tune_hyperparameters(
    data_dir: str,
    project_name: str = "track_prediction_tuning",
    max_epochs: int = 50,
    factor: int = 3,
    hyperband_iterations: int = 1,
    batch_size: int = 32,
    distributed: bool = False,
    executions_per_trial: int = 2,
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
        )

    print(
        f"Starting Hyperband hyperparameter tuning with max_epochs={max_epochs}, factor={factor}..."
    )
    print("Search space:")
    tuner.search_space_summary()

    # Setup callbacks for tuning
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True, verbose=0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=0
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=f"{project_name}_best.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=0,
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
        keras.callbacks.ModelCheckpoint(
            filepath=f"{model_save_path}_best.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        TqdmCallback(verbose=1, epochs_desc="Training Progress", steps_desc="Step"),
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

    model.save(f"{model_save_path}_final.keras")
    print(f"Model saved to {model_save_path}_final.keras")

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
    val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)
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

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=actual_patience,
            restore_best_weights=True,
            verbose=1,
            min_delta=1e-4,  # Smaller threshold for small datasets
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=round((2 + actual_patience) / 2),
            min_lr=1e-7,
            verbose=1,
            min_delta=1e-4,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=f"{model_save_path}_best.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        # Cosine annealing for better convergence
        keras.callbacks.LearningRateScheduler(
            lambda epoch: 0.001 * (0.5 * (1 + np.cos(np.pi * epoch / 50))), verbose=0
        ),
    ]

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
    callbacks.append(
        TqdmCallback(verbose=1, epochs_desc="Training Progress", steps_desc="Step")
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

    # Save final model
    model.save(f"{model_save_path}_final.keras")
    print(f"Model saved to {model_save_path}_final.keras")

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

    # Evaluate model
    print("\nFinal evaluation:")
    val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)
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
            "adaptive_learning_rate": adaptive_lr,
            "use_class_weights": use_class_weights,
            "dataset_size": dataset_size,
        }

        final_metrics = {
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
            "total_epochs_trained": len(history.history["loss"]),
        }

        # Add best metrics from history
        if "val_accuracy" in history.history:
            final_metrics["best_validation_accuracy"] = max(
                history.history["val_accuracy"]
            )
            final_metrics["best_validation_loss"] = min(history.history["val_loss"])

        additional_info = {
            "model_parameters": model.count_params(),
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

    return history


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

    trained_models = []
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
            f"{model_save_path}_model_{i}", dataset_size, epochs=epochs
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
        val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)
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
        model.save(f"{model_save_path}_model_{i}_final.keras")
        trained_models.append(model)
        tqdm.write(f"Model {i + 1} completed - Accuracy: {val_accuracy:.4f}")

    # Evaluate ensemble
    print("\nEvaluating ensemble performance...")
    ensemble_predictions = []

    for model in tqdm(
        trained_models, desc="Evaluating ensemble models", unit="model", leave=False
    ):
        predictions = model.predict(val_ds, verbose=0)
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


def _collect_dataset_data(dataset: tf.data.Dataset) -> tuple[dict[str, Any], Any]:
    """Collect all data from a tf.data.Dataset into numpy arrays."""
    all_features: dict[str, list[Any]] = {
        "station": [],
        "route": [],
        "direction": [],
        "hour_sin": [],
        "hour_cos": [],
        "day_sin": [],
        "day_cos": [],
        "month_sin": [],
        "month_cos": [],
    }
    all_targets: list[Any] = []

    for features, targets in dataset:
        # Handle batched data
        for key in all_features.keys():
            if key in features:
                all_features[key].extend(features[key].numpy())
        all_targets.extend(targets.numpy())

    # Convert to numpy arrays
    features_arrays: dict[str, Any] = {}
    for key in all_features.keys():
        features_arrays[key] = np.array(all_features[key])
    targets_array = np.array(all_targets)

    return features_arrays, targets_array


def _create_dataset_from_arrays(
    features: dict[str, Any],
    targets: Any,
    indices: Any,
    batch_size: int = 32,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Create a tf.data.Dataset from numpy arrays using specified indices."""
    # Extract data for specified indices
    fold_features = {}
    for key, values in features.items():
        fold_features[key] = values[indices]
    fold_targets = targets[indices]

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((fold_features, fold_targets))

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(indices), reshuffle_each_iteration=True
        )

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _calculate_cv_statistics(scores: list[float]) -> dict[str, float]:
    """Calculate comprehensive statistics for cross-validation scores."""
    scores_array = np.array(scores)

    return {
        "mean": np.mean(scores_array),
        "std": np.std(scores_array),
        "min": np.min(scores_array),
        "max": np.max(scores_array),
        "median": np.median(scores_array),
        "q1": np.percentile(scores_array, 25),
        "q3": np.percentile(scores_array, 75),
        "iqr": np.percentile(scores_array, 75) - np.percentile(scores_array, 25),
        "cv_coefficient": np.std(scores_array) / np.mean(scores_array)
        if np.mean(scores_array) != 0
        else 0,
        "confidence_interval_95": np.std(scores_array)
        * 1.96
        / np.sqrt(len(scores_array)),
        "sem": np.std(scores_array)
        / np.sqrt(len(scores_array)),  # Standard error of mean
    }


def _evaluate_fold_predictions(
    y_true: Any,
    y_pred: Any,
    y_proba: Any,
    vocab_info: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate predictions for a single fold with comprehensive metrics."""
    from sklearn.metrics import (
        accuracy_score,
        log_loss,
        precision_recall_fscore_support,
        top_k_accuracy_score,
    )

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Top-k accuracy
    top_3_acc = (
        top_k_accuracy_score(y_true, y_proba, k=3)
        if y_proba.shape[1] >= 3
        else accuracy
    )
    top_5_acc = (
        top_k_accuracy_score(y_true, y_proba, k=5)
        if y_proba.shape[1] >= 5
        else accuracy
    )

    # Log loss
    try:
        logloss = log_loss(y_true, y_proba)
    except ValueError:
        logloss = float("inf")

    # Per-class metrics
    per_class_precision, per_class_recall, per_class_f1, per_class_support = (
        precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "top_3_accuracy": top_3_acc,
        "top_5_accuracy": top_5_acc,
        "log_loss": logloss,
        "per_class_metrics": {
            "precision": per_class_precision.tolist(),
            "recall": per_class_recall.tolist(),
            "f1_score": per_class_f1.tolist(),
            "support": per_class_support.tolist(),
        },
    }


def evaluate_with_cross_validation(
    data_dir: str,
    k_folds: int = 5,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    model_save_path: str = "cv_evaluation",
    stratified: bool = True,
    save_fold_models: bool = False,
    random_state: int = 42,
    generate_report_func=None,
) -> dict[str, Any]:
    """
    Evaluate model performance using proper k-fold cross-validation.

    Args:
        data_dir: Directory containing TFRecord files
        k_folds: Number of folds for cross-validation
        epochs: Number of epochs to train each fold
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        model_save_path: Path prefix for saving models
        stratified: Whether to use stratified k-fold (maintains class distribution)
        save_fold_models: Whether to save individual fold models
        random_state: Random state for reproducible splits
        generate_report_func: Optional function to generate training reports

    Returns:
        Dictionary containing comprehensive cross-validation results
    """
    start_time = time.time()

    print(f"Loading dataset for {k_folds}-fold cross-validation...")
    train_ds, val_ds, vocab_info = load_tfrecord_dataset(
        data_dir,
        batch_size=1,
        augment_data=False,  # Use batch_size=1 for data collection
    )

    print("Collecting all data for proper cross-validation splits...")

    # Collect all data from both train and validation sets
    combined_ds = train_ds.concatenate(val_ds)
    all_features, all_targets = _collect_dataset_data(combined_ds)

    total_samples = len(all_targets)
    print(f"Total samples collected: {total_samples}")
    print("Feature shapes: {k: v.shape for k, v in all_features.items()}")
    print(f"Target shape: {all_targets.shape}")
    print(f"Unique classes: {len(np.unique(all_targets))}")

    # Set up cross-validation splitter
    if stratified and len(np.unique(all_targets)) > 1:
        cv_splitter = StratifiedKFold(
            n_splits=k_folds, shuffle=True, random_state=random_state
        )
        print(f"Using stratified {k_folds}-fold cross-validation")
    else:
        cv_splitter = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
        print(f"Using standard {k_folds}-fold cross-validation")

    # Store results for each fold
    fold_results = []
    fold_histories = []
    fold_predictions = []

    # Aggregate metrics across folds
    all_fold_metrics: dict[str, list[Any]] = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "top_3_accuracy": [],
        "top_5_accuracy": [],
        "log_loss": [],
        "training_time": [],
        "epochs_trained": [],
        "final_train_loss": [],
        "final_val_loss": [],
    }

    for fold_idx, (train_indices, val_indices) in enumerate(
        tqdm(
            cv_splitter.split(all_targets, all_targets if stratified else None),
            total=k_folds,
            desc="Cross-validation folds",
            unit="fold",
        )
    ):
        fold_start_time = time.time()
        tqdm.write(f"\nFold {fold_idx + 1}/{k_folds}")
        tqdm.write(
            f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}"
        )

        # Create datasets for this fold
        train_fold_ds = _create_dataset_from_arrays(
            all_features, all_targets, train_indices, batch_size, shuffle=True
        )
        val_fold_ds = _create_dataset_from_arrays(
            all_features, all_targets, val_indices, batch_size, shuffle=False
        )

        # Calculate steps per epoch
        train_steps = len(train_indices) // batch_size
        val_steps = len(val_indices) // batch_size

        # Create and compile model for this fold
        model = create_simple_model(vocab_info, use_regularization=True)

        # Use different learning rates for diversity (similar to ensemble)
        fold_lr = learning_rate * (
            0.8 + 0.4 * np.random.RandomState(random_state + fold_idx).random()
        )
        optimizer = keras.optimizers.Adam(learning_rate=fold_lr)

        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy", "top_k_categorical_accuracy"],
        )

        # Setup callbacks with fold-specific paths
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=1,
                min_delta=1e-4,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1
            ),
        ]

        if save_fold_models:
            callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    filepath=f"{model_save_path}_fold_{fold_idx + 1}_best.keras",
                    monitor="val_accuracy",
                    save_best_only=True,
                    verbose=1,
                )
            )

        # Train model for this fold
        tqdm.write(f"Training fold {fold_idx + 1} with learning rate {fold_lr:.6f}...")

        history = model.fit(
            train_fold_ds,
            validation_data=val_fold_ds,
            epochs=epochs,
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            callbacks=callbacks,
            verbose=1,
        )

        # Get predictions for this fold
        val_predictions_proba = model.predict(val_fold_ds, verbose=0)
        val_predictions = np.argmax(val_predictions_proba, axis=1)
        val_true = all_targets[val_indices]

        # Evaluate fold performance
        fold_metrics = _evaluate_fold_predictions(
            val_true, val_predictions, val_predictions_proba, vocab_info
        )

        # Calculate fold training time
        fold_training_time = time.time() - fold_start_time

        # Store fold results
        fold_result = {
            "fold": fold_idx + 1,
            "train_size": len(train_indices),
            "val_size": len(val_indices),
            "epochs_trained": len(history.history["loss"]),
            "training_time": fold_training_time,
            "learning_rate": fold_lr,
            "final_train_loss": history.history["loss"][-1],
            "final_val_loss": history.history["val_loss"][-1],
            "best_val_accuracy": max(history.history["val_accuracy"]),
            "best_val_loss": min(history.history["val_loss"]),
            "model_parameters": model.count_params(),
            **fold_metrics,
        }

        fold_results.append(fold_result)
        fold_histories.append(history.history)
        fold_predictions.append(
            {
                "true": val_true,
                "predicted": val_predictions,
                "probabilities": val_predictions_proba,
            }
        )

        # Update aggregate metrics
        for metric in all_fold_metrics:
            if metric in fold_result:
                all_fold_metrics[metric].append(fold_result[metric])

        # Save individual fold model if requested
        if save_fold_models:
            model.save(f"{model_save_path}_fold_{fold_idx + 1}_final.keras")

        tqdm.write(f"Fold {fold_idx + 1} completed:")
        tqdm.write(f"  Accuracy: {fold_metrics['accuracy']:.4f}")
        tqdm.write(f"  F1-Score: {fold_metrics['f1_score']:.4f}")
        tqdm.write(f"  Top-3 Accuracy: {fold_metrics['top_3_accuracy']:.4f}")
        tqdm.write(f"  Training time: {fold_training_time:.2f}s")

    # Calculate comprehensive statistics
    cv_statistics = {}
    for metric, values in all_fold_metrics.items():
        if values:  # Only calculate stats for metrics that have values
            cv_statistics[metric] = _calculate_cv_statistics(values)

    total_cv_time = time.time() - start_time

    # Print comprehensive results
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 80)
    print(f"Strategy: {'Stratified' if stratified else 'Standard'} {k_folds}-fold CV")
    print(f"Total CV time: {total_cv_time:.2f}s")
    print(f"Average training time per fold: {total_cv_time / k_folds:.2f}s")
    print()

    for metric, stats in cv_statistics.items():
        if metric in [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "top_3_accuracy",
            "top_5_accuracy",
        ]:
            print(f"{metric.replace('_', ' ').title()}:")
            print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"  95% CI: ±{stats['confidence_interval_95']:.4f}")
            print()

    # Prepare comprehensive results
    cv_results = {
        "cv_statistics": cv_statistics,
        "fold_results": fold_results,
        "fold_histories": fold_histories,
        "fold_predictions": fold_predictions,
        "cv_config": {
            "k_folds": k_folds,
            "stratified": stratified,
            "random_state": random_state,
            "total_samples": total_samples,
            "epochs_per_fold": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        },
        "timing": {
            "total_cv_time": total_cv_time,
            "average_fold_time": total_cv_time / k_folds,
        },
    }

    # Generate cross-validation report if function provided
    if generate_report_func:
        training_params = {
            "k_folds": k_folds,
            "stratified": stratified,
            "epochs_per_fold": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "dataset_size": total_samples,
            "random_state": random_state,
        }

        final_metrics = {
            "mean_cv_accuracy": cv_statistics["accuracy"]["mean"],
            "std_cv_accuracy": cv_statistics["accuracy"]["std"],
            "cv_accuracy_95_ci": cv_statistics["accuracy"]["confidence_interval_95"],
            "best_fold_accuracy": cv_statistics["accuracy"]["max"],
            "worst_fold_accuracy": cv_statistics["accuracy"]["min"],
            "mean_f1_score": cv_statistics["f1_score"]["mean"],
            "mean_top3_accuracy": cv_statistics["top_3_accuracy"]["mean"],
            "cv_coefficient_variation": cv_statistics["accuracy"]["cv_coefficient"],
        }

        additional_info = {
            "fold_details": fold_results,
            "cv_strategy": f"{'Stratified' if stratified else 'Standard'} K-Fold",
            "total_models_trained": k_folds,
            "save_individual_models": save_fold_models,
            "learning_rate_variation": "Applied per fold for robustness",
            "early_stopping": "Applied with patience=5",
            "statistical_significance": f"95% CI: ±{cv_statistics['accuracy']['confidence_interval_95']:.4f}",
        }

        generate_report_func(
            "cv",
            model_save_path,
            vocab_info,
            training_params,
            final_metrics,
            total_cv_time,
            additional_info,
        )

    return cv_results


def compare_cv_strategies(
    data_dir: str,
    strategies: list[dict[str, Any]] | None = None,
    model_save_path: str = "cv_comparison",
    generate_report_func=None,
) -> dict[str, Any]:
    """
    Compare different cross-validation strategies and hyperparameters.

    Args:
        data_dir: Directory containing TFRecord files
        strategies: List of CV strategy configurations to compare
        model_save_path: Path prefix for saving comparison results
        generate_report_func: Optional function to generate reports

    Returns:
        Dictionary containing comparison results across strategies
    """
    if strategies is None:
        strategies = [
            {
                "k_folds": 3,
                "stratified": True,
                "epochs": 20,
                "name": "3-fold Stratified",
            },
            {
                "k_folds": 5,
                "stratified": True,
                "epochs": 30,
                "name": "5-fold Stratified",
            },
            {
                "k_folds": 5,
                "stratified": False,
                "epochs": 30,
                "name": "5-fold Standard",
            },
            {
                "k_folds": 10,
                "stratified": True,
                "epochs": 25,
                "name": "10-fold Stratified",
            },
        ]

    print("Comparing cross-validation strategies...")
    print(f"Strategies to compare: {len(strategies)}")

    comparison_results = {}
    strategy_summaries = []

    for i, strategy in enumerate(strategies):
        strategy_name = strategy.get("name", f"Strategy_{i + 1}")
        print(f"\n{'=' * 60}")
        print(f"Running strategy: {strategy_name}")
        print(f"Configuration: {strategy}")
        print("=" * 60)

        # Extract strategy parameters
        strategy_params = {k: v for k, v in strategy.items() if k != "name"}

        # Run cross-validation with this strategy
        cv_results = evaluate_with_cross_validation(
            data_dir=data_dir,
            model_save_path=f"{model_save_path}_{strategy_name.replace(' ', '_').lower()}",
            generate_report_func=None,  # Don't generate individual reports
            **strategy_params,
        )

        # Store results
        comparison_results[strategy_name] = cv_results

        # Create summary for comparison
        acc_stats = cv_results["cv_statistics"]["accuracy"]
        strategy_summaries.append(
            {
                "strategy": strategy_name,
                "config": strategy,
                "mean_accuracy": acc_stats["mean"],
                "std_accuracy": acc_stats["std"],
                "accuracy_95_ci": acc_stats["confidence_interval_95"],
                "cv_coefficient": acc_stats["cv_coefficient"],
                "total_time": cv_results["timing"]["total_cv_time"],
                "avg_fold_time": cv_results["timing"]["average_fold_time"],
                "robustness_score": 1
                / (
                    1 + acc_stats["cv_coefficient"]
                ),  # Lower variation = higher robustness
            }
        )

    # Find best strategy
    best_strategy = max(strategy_summaries, key=lambda x: x["mean_accuracy"])
    most_robust = max(strategy_summaries, key=lambda x: x["robustness_score"])
    fastest = min(strategy_summaries, key=lambda x: x["total_time"])

    # Print comparison summary
    print(f"\n{'=' * 80}")
    print("CROSS-VALIDATION STRATEGY COMPARISON")
    print("=" * 80)

    print(
        f"{'Strategy':<25} {'Accuracy':<15} {'Std Dev':<10} {'95% CI':<10} {'Time (s)':<10}"
    )
    print("-" * 80)
    for summary in strategy_summaries:
        print(
            f"{summary['strategy']:<25} "
            f"{summary['mean_accuracy']:.4f}±{summary['std_accuracy']:.4f}  "
            f"{summary['std_accuracy']:.4f}     "
            f"±{summary['accuracy_95_ci']:.4f}    "
            f"{summary['total_time']:.1f}"
        )

    print(
        f"\nBest accuracy: {best_strategy['strategy']} ({best_strategy['mean_accuracy']:.4f})"
    )
    print(
        f"Most robust: {most_robust['strategy']} (CV coefficient: {most_robust['cv_coefficient']:.4f})"
    )
    print(f"Fastest: {fastest['strategy']} ({fastest['total_time']:.1f}s)")

    comparison_summary = {
        "strategies_compared": strategy_summaries,
        "best_accuracy": best_strategy,
        "most_robust": most_robust,
        "fastest": fastest,
        "detailed_results": comparison_results,
    }

    # Generate comparison report if function provided
    if generate_report_func:
        # Use the best strategy's vocab_info for the report
        vocab_info = {"metadata": {"total_records": "comparison"}}  # Placeholder

        training_params = {
            "comparison_type": "cross_validation_strategies",
            "strategies_count": len(strategies),
            "best_strategy": best_strategy["strategy"],
            "strategies_tested": [s["name"] for s in strategies],
        }

        final_metrics = {
            "best_mean_accuracy": best_strategy["mean_accuracy"],
            "best_std_accuracy": best_strategy["std_accuracy"],
            "most_robust_strategy": most_robust["strategy"],
            "robustness_score": most_robust["robustness_score"],
            "fastest_strategy": fastest["strategy"],
            "fastest_time": fastest["total_time"],
        }

        total_comparison_time = sum(s["total_time"] for s in strategy_summaries)

        additional_info = {
            "strategy_summaries": strategy_summaries,
            "recommendation": f"Best overall: {best_strategy['strategy']}",
            "robustness_analysis": f"Most consistent: {most_robust['strategy']}",
            "efficiency_analysis": f"Most efficient: {fastest['strategy']}",
            "total_comparison_time": total_comparison_time,
        }

        generate_report_func(
            "cv_comparison",
            model_save_path,
            vocab_info,
            training_params,
            final_metrics,
            total_comparison_time,
            additional_info,
        )

    return comparison_summary


def analyze_cv_results(
    cv_results: dict[str, Any],
    model_save_path: str = "cv_analysis",
    save_detailed_analysis: bool = True,
) -> dict[str, Any]:
    """
    Analyze cross-validation results in detail and provide insights.

    Args:
        cv_results: Results from evaluate_with_cross_validation
        model_save_path: Path prefix for saving analysis files
        save_detailed_analysis: Whether to save detailed analysis to files

    Returns:
        Dictionary containing detailed analysis and insights
    """

    print("Analyzing cross-validation results...")

    # Extract data for analysis
    fold_results = cv_results["fold_results"]
    cv_statistics = cv_results["cv_statistics"]
    cv_config = cv_results["cv_config"]

    # Analysis components
    analysis: dict[str, Any] = {
        "performance_summary": {},
        "consistency_analysis": {},
        "convergence_analysis": {},
        "efficiency_analysis": {},
        "recommendations": [],
        "outlier_analysis": {},
        "learning_curve_insights": {},
    }

    # Performance Summary
    acc_stats = cv_statistics["accuracy"]
    analysis["performance_summary"] = {
        "overall_performance": "Excellent"
        if acc_stats["mean"] > 0.9
        else "Good"
        if acc_stats["mean"] > 0.8
        else "Fair"
        if acc_stats["mean"] > 0.7
        else "Poor",
        "mean_accuracy": acc_stats["mean"],
        "performance_range": acc_stats["max"] - acc_stats["min"],
        "best_fold": max(fold_results, key=lambda x: x["accuracy"])["fold"],
        "worst_fold": min(fold_results, key=lambda x: x["accuracy"])["fold"],
        "performance_distribution": {
            "q1": acc_stats["q1"],
            "median": acc_stats["median"],
            "q3": acc_stats["q3"],
            "iqr": acc_stats["iqr"],
        },
    }

    # Consistency Analysis
    cv_coeff = acc_stats["cv_coefficient"]
    analysis["consistency_analysis"] = {
        "consistency_rating": "Excellent"
        if cv_coeff < 0.05
        else "Good"
        if cv_coeff < 0.1
        else "Fair"
        if cv_coeff < 0.15
        else "Poor",
        "cv_coefficient": cv_coeff,
        "consistency_interpretation": "Very stable"
        if cv_coeff < 0.05
        else "Stable"
        if cv_coeff < 0.1
        else "Moderate variation"
        if cv_coeff < 0.15
        else "High variation - investigate data or model",
        "fold_variability": acc_stats["std"],
        "confidence_bounds": {
            "lower": acc_stats["mean"] - acc_stats["confidence_interval_95"],
            "upper": acc_stats["mean"] + acc_stats["confidence_interval_95"],
        },
    }

    # Convergence Analysis
    epochs_trained = [fold["epochs_trained"] for fold in fold_results]
    total_epochs_possible = cv_config["epochs_per_fold"]

    analysis["convergence_analysis"] = {
        "average_epochs_needed": np.mean(epochs_trained),
        "early_stopping_rate": sum(
            1 for e in epochs_trained if e < total_epochs_possible
        )
        / len(epochs_trained),
        "convergence_consistency": np.std(epochs_trained),
        "fastest_convergence": min(epochs_trained),
        "slowest_convergence": max(epochs_trained),
        "convergence_interpretation": "Fast and consistent"
        if np.std(epochs_trained) < 3
        else "Moderate"
        if np.std(epochs_trained) < 6
        else "Inconsistent - check hyperparameters",
    }

    # Efficiency Analysis
    training_times = cv_statistics["training_time"]
    analysis["efficiency_analysis"] = {
        "average_training_time": training_times["mean"],
        "total_cv_time": cv_results["timing"]["total_cv_time"],
        "time_per_epoch": training_times["mean"] / np.mean(epochs_trained),
        "efficiency_rating": "Excellent"
        if training_times["mean"] < 60
        else "Good"
        if training_times["mean"] < 300
        else "Fair"
        if training_times["mean"] < 600
        else "Slow",
        "time_consistency": training_times["cv_coefficient"],
        "projected_full_training_time": training_times["mean"] * total_epochs_possible,
    }

    # Outlier Analysis
    accuracies = [fold["accuracy"] for fold in fold_results]
    q1, q3 = np.percentile(accuracies, [25, 75])
    iqr = q3 - q1
    outlier_threshold_low = q1 - 1.5 * iqr
    outlier_threshold_high = q3 + 1.5 * iqr

    outliers = []
    for fold in fold_results:
        if (
            fold["accuracy"] < outlier_threshold_low
            or fold["accuracy"] > outlier_threshold_high
        ):
            outliers.append(
                {
                    "fold": fold["fold"],
                    "accuracy": fold["accuracy"],
                    "type": "low"
                    if fold["accuracy"] < outlier_threshold_low
                    else "high",
                }
            )

    analysis["outlier_analysis"] = {
        "outliers_detected": len(outliers),
        "outlier_folds": outliers,
        "outlier_impact": "Significant"
        if len(outliers) > len(fold_results) * 0.2
        else "Moderate"
        if len(outliers) > 0
        else "None",
    }

    # Learning Curve Insights (if histories available)
    if "fold_histories" in cv_results:
        val_losses_final = [
            min(hist["val_loss"]) for hist in cv_results["fold_histories"]
        ]
        train_losses_final = [hist["loss"][-1] for hist in cv_results["fold_histories"]]

        overfitting_indicators = []
        for i, (train_loss, val_loss) in enumerate(
            zip(train_losses_final, val_losses_final)
        ):
            gap = val_loss - train_loss
            if gap > 0.5:  # Significant gap indicating overfitting
                overfitting_indicators.append({"fold": i + 1, "gap": gap})

        analysis["learning_curve_insights"] = {
            "average_final_val_loss": np.mean(val_losses_final),
            "average_final_train_loss": np.mean(train_losses_final),
            "average_generalization_gap": np.mean(val_losses_final)
            - np.mean(train_losses_final),
            "overfitting_folds": overfitting_indicators,
            "overfitting_assessment": "Significant"
            if len(overfitting_indicators) > len(fold_results) * 0.3
            else "Moderate"
            if len(overfitting_indicators) > 0
            else "Minimal",
        }

    # Generate Recommendations
    recommendations = []

    # Performance recommendations
    if acc_stats["mean"] < 0.8:
        recommendations.append(
            "Consider increasing model complexity or training epochs"
        )
    if (
        acc_stats["mean"] > 0.95
        and analysis["learning_curve_insights"].get("overfitting_assessment")
        == "Significant"
    ):
        recommendations.append("Model may be overfitting - consider regularization")

    # Consistency recommendations
    if cv_coeff > 0.15:
        recommendations.append(
            "High variance detected - consider stratified sampling or larger dataset"
        )
    if cv_coeff > 0.1:
        recommendations.append("Consider ensemble methods to improve stability")

    # Convergence recommendations
    if analysis["convergence_analysis"]["early_stopping_rate"] < 0.3:
        recommendations.append(
            "Most folds used full epochs - consider increasing epoch limit"
        )
    if analysis["convergence_analysis"]["early_stopping_rate"] > 0.8:
        recommendations.append(
            "Frequent early stopping - consider reducing epochs or adjusting patience"
        )

    # Efficiency recommendations
    if training_times["mean"] > 300:
        recommendations.append(
            "Training time is high - consider model optimization or smaller batch sizes"
        )

    # Outlier recommendations
    if len(outliers) > 0:
        recommendations.append(
            f"Investigate fold(s) {[o['fold'] for o in outliers]} for data quality issues"
        )

    analysis["recommendations"] = recommendations

    # Print analysis summary
    print(f"\n{'=' * 80}")
    print("CROSS-VALIDATION ANALYSIS SUMMARY")
    print("=" * 80)

    print(
        f"Overall Performance: {analysis['performance_summary']['overall_performance']}"
    )
    print(
        f"Mean Accuracy: {acc_stats['mean']:.4f} (95% CI: ±{acc_stats['confidence_interval_95']:.4f})"
    )
    print(
        f"Consistency: {analysis['consistency_analysis']['consistency_rating']} (CV: {cv_coeff:.4f})"
    )
    print(
        f"Convergence: {analysis['convergence_analysis']['convergence_interpretation']}"
    )
    print(f"Efficiency: {analysis['efficiency_analysis']['efficiency_rating']}")

    if outliers:
        print(f"Outliers Detected: {len(outliers)} fold(s)")

    if recommendations:
        print("\nRecommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")

    # Save detailed analysis if requested
    if save_detailed_analysis:
        analysis_file = f"{model_save_path}_detailed_analysis.json"
        with open(analysis_file, "w") as f:
            # Convert numpy types to Python native types for JSON serialization
            def convert_numpy(obj: Any) -> Any:
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj

            def deep_convert(obj: Any) -> Any:
                if isinstance(obj, dict):
                    return {k: deep_convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [deep_convert(v) for v in obj]
                else:
                    return convert_numpy(obj)

            json.dump(deep_convert(analysis), f, indent=2)
        print(f"\nDetailed analysis saved to: {analysis_file}")

        # Save fold-by-fold comparison
        fold_comparison_file = f"{model_save_path}_fold_comparison.json"
        fold_summary = []
        for fold in fold_results:
            fold_summary.append(
                {
                    "fold": fold["fold"],
                    "accuracy": fold["accuracy"],
                    "f1_score": fold["f1_score"],
                    "epochs_trained": fold["epochs_trained"],
                    "training_time": fold["training_time"],
                    "learning_rate": fold["learning_rate"],
                }
            )

        with open(fold_comparison_file, "w") as f:
            json.dump(fold_summary, f, indent=2)
        print(f"Fold comparison saved to: {fold_comparison_file}")

    return analysis
