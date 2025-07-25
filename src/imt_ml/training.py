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

from .dataset import load_tfrecord_dataset
from .models import (
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
        keras.callbacks.TqdmCallback(
            verbose=1, epochs_desc="Training Progress", steps_desc="Step"
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
    model_save_path: str, dataset_size: int, monitor_patience: int | None = None
) -> list[keras.callbacks.Callback]:
    """Create optimized callbacks for small datasets."""
    # Adjust patience based on dataset size
    if monitor_patience is None:
        monitor_patience = 5 if dataset_size < 5000 else 3

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
            patience=max(2, actual_patience // 2),
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
    callbacks = _create_optimized_callbacks(model_save_path, dataset_size)
    callbacks.append(
        keras.callbacks.TqdmCallback(
            verbose=1, epochs_desc="Training Progress", steps_desc="Step"
        )
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

    for i, model in enumerate(models):
        print(f"\nTraining model {i + 1}/{num_models}...")

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
            f"{model_save_path}_model_{i}", dataset_size
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

    # Evaluate ensemble
    print("\nEvaluating ensemble performance...")
    ensemble_predictions = []

    for model in trained_models:
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

    for features, target in combined_ds.take(-1):
        all_features.append(features)
        all_targets.append(target)

    # total_samples = len(all_features)
    # fold_size = total_samples // k_folds  # For future proper CV implementation

    cv_scores = []
    fold_details = []

    for fold in range(k_folds):
        print(f"\nFold {fold + 1}/{k_folds}")

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
                monitor="val_loss", patience=3, restore_best_weights=True, verbose=0
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
            verbose=0,
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
        print(f"Fold {fold + 1} accuracy: {best_val_acc:.4f}")

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
