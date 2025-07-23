#!/usr/bin/env python3
"""
TensorFlow helper functions for reading and preprocessing the exported track data.

This module provides utilities to read TFRecord files and create TensorFlow datasets
for training track prediction models.
"""

import json
from pathlib import Path
from typing import Any, Callable

import keras
import keras_tuner as kt
import tensorflow as tf


def parse_tfrecord_fn(example_proto: tf.Tensor) -> dict[str, Any]:
    """Parse a TFRecord example."""
    feature_description = {
        "station_id": tf.io.FixedLenFeature([], tf.string),
        "route_id": tf.io.FixedLenFeature([], tf.string),
        "trip_id": tf.io.FixedLenFeature([], tf.string),
        "headsign": tf.io.FixedLenFeature([], tf.string),
        "direction_id": tf.io.FixedLenFeature([], tf.int64),
        "assignment_type": tf.io.FixedLenFeature([], tf.string),
        "track_number": tf.io.FixedLenFeature([], tf.string),
        "scheduled_timestamp": tf.io.FixedLenFeature([], tf.float32),
        "actual_timestamp": tf.io.FixedLenFeature([], tf.float32),
        "recorded_timestamp": tf.io.FixedLenFeature([], tf.float32),
        "day_of_week": tf.io.FixedLenFeature([], tf.int64),
        "hour": tf.io.FixedLenFeature([], tf.int64),
        "minute": tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    return parsed  # type: ignore[no-any-return]


def create_feature_engineering_fn(
    station_vocab: list[str], route_vocab: list[str], track_vocab: list[str]
) -> Callable[[dict[str, tf.Tensor]], tuple[dict[str, tf.Tensor], tf.Tensor]]:
    """Create a feature engineering function with vocabularies."""

    # Create lookup tables
    station_table = tf.lookup.StaticVocabularyTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=station_vocab,
            values=list(range(len(station_vocab))),
            key_dtype=tf.string,
            value_dtype=tf.int64,
        ),
        num_oov_buckets=1,  # For unknown stations
    )

    route_table = tf.lookup.StaticVocabularyTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=route_vocab,
            values=list(range(len(route_vocab))),
            key_dtype=tf.string,
            value_dtype=tf.int64,
        ),
        num_oov_buckets=1,
    )

    track_table = tf.lookup.StaticVocabularyTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=track_vocab,
            values=list(range(len(track_vocab))),
            key_dtype=tf.string,
            value_dtype=tf.int64,
        ),
        num_oov_buckets=1,
    )

    def feature_engineering(
        example: dict[str, tf.Tensor],
    ) -> tuple[dict[str, tf.Tensor], tf.Tensor]:
        """Apply feature engineering to a parsed example."""
        # Convert categorical features to indices
        station_idx = station_table.lookup(example["station_id"])
        route_idx = route_table.lookup(example["route_id"])
        track_idx = track_table.lookup(example["track_number"])

        # Time-based features
        hour_sin = tf.sin(2 * 3.14159 * tf.cast(example["hour"], tf.float32) / 24.0)
        hour_cos = tf.cos(2 * 3.14159 * tf.cast(example["hour"], tf.float32) / 24.0)
        minute_sin = tf.sin(2 * 3.14159 * tf.cast(example["minute"], tf.float32) / 60.0)
        minute_cos = tf.cos(2 * 3.14159 * tf.cast(example["minute"], tf.float32) / 60.0)
        day_sin = tf.sin(
            2 * 3.14159 * tf.cast(example["day_of_week"], tf.float32) / 7.0
        )
        day_cos = tf.cos(
            2 * 3.14159 * tf.cast(example["day_of_week"], tf.float32) / 7.0
        )

        # Create features dict
        features: dict[str, tf.Tensor] = {
            "station_id": station_idx,
            "route_id": route_idx,
            "direction_id": example["direction_id"],
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "minute_sin": minute_sin,
            "minute_cos": minute_cos,
            "day_sin": day_sin,
            "day_cos": day_cos,
            "scheduled_timestamp": example["scheduled_timestamp"],
        }

        # Target (track number as index)
        target: tf.Tensor = track_idx

        return features, target

    return feature_engineering


def load_tfrecord_dataset(
    data_dir: str,
    batch_size: int = 32,
    train_split: float = 0.8,
    shuffle_buffer: int = 10000,
) -> tuple[tf.data.Dataset, tf.data.Dataset, dict[str, Any]]:
    """Load TFRecord dataset and create train/validation splits."""

    data_path = Path(data_dir)

    # Load metadata
    metadata_path = data_path / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path) as f:
        metadata = json.load(f)

    # Find all TFRecord files
    tfrecord_files = list(data_path.glob("*.tfrecord"))
    if not tfrecord_files:
        raise FileNotFoundError(f"No TFRecord files found in {data_path}")

    print(f"Found {len(tfrecord_files)} TFRecord files")

    # Create dataset
    dataset = tf.data.TFRecordDataset([str(f) for f in tfrecord_files])
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # Build vocabularies by scanning the dataset
    print("Building vocabularies...")
    stations = set()
    routes = set()
    tracks = set()

    for example in dataset.take(-1):  # Scan all examples
        stations.add(example["station_id"].numpy().decode("utf-8"))
        routes.add(example["route_id"].numpy().decode("utf-8"))
        track_str = example["track_number"].numpy().decode("utf-8")
        if track_str:  # Only add non-empty tracks
            tracks.add(track_str)

    station_vocab = sorted(list(stations))
    route_vocab = sorted(list(routes))
    track_vocab = sorted(list(tracks))

    print(
        f"Vocabularies: {len(station_vocab)} stations, {len(route_vocab)} routes, {len(track_vocab)} tracks"
    )

    # Apply feature engineering
    feature_fn = create_feature_engineering_fn(station_vocab, route_vocab, track_vocab)
    dataset = dataset.map(feature_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # Filter out examples with empty track numbers (target = vocab_size means OOV/empty)
    dataset = dataset.filter(lambda x, y: y < len(track_vocab))

    # Split train/validation
    total_size = metadata["total_records"]
    train_size = int(total_size * train_split)

    # Shuffle before split
    dataset = dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=False)

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)

    # Batch and prefetch with repeat for small datasets
    train_dataset = train_dataset.repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Calculate steps per epoch
    train_steps_per_epoch = max(1, train_size // batch_size)
    val_steps_per_epoch = max(1, (total_size - train_size) // batch_size)

    # Create vocab info for model building
    vocab_info = {
        "station_vocab": station_vocab,
        "route_vocab": route_vocab,
        "track_vocab": track_vocab,
        "num_stations": len(station_vocab),
        "num_routes": len(route_vocab),
        "num_tracks": len(track_vocab),
        "metadata": metadata,
        "train_steps_per_epoch": train_steps_per_epoch,
        "val_steps_per_epoch": val_steps_per_epoch,
    }

    return train_dataset, val_dataset, vocab_info


def create_simple_model(vocab_info: dict[str, Any]) -> Any:
    """Create a simple neural network for track prediction."""

    # Input layers
    station_input: tf.Tensor = keras.layers.Input(
        shape=(), name="station_id", dtype=tf.int64
    )
    route_input: tf.Tensor = keras.layers.Input(
        shape=(), name="route_id", dtype=tf.int64
    )
    direction_input: tf.Tensor = keras.layers.Input(
        shape=(), name="direction_id", dtype=tf.int64
    )

    # Time features
    hour_sin_input: tf.Tensor = keras.layers.Input(
        shape=(), name="hour_sin", dtype=tf.float32
    )
    hour_cos_input: tf.Tensor = keras.layers.Input(
        shape=(), name="hour_cos", dtype=tf.float32
    )
    minute_sin_input: tf.Tensor = keras.layers.Input(
        shape=(), name="minute_sin", dtype=tf.float32
    )
    minute_cos_input: tf.Tensor = keras.layers.Input(
        shape=(), name="minute_cos", dtype=tf.float32
    )
    day_sin_input: tf.Tensor = keras.layers.Input(
        shape=(), name="day_sin", dtype=tf.float32
    )
    day_cos_input: tf.Tensor = keras.layers.Input(
        shape=(), name="day_cos", dtype=tf.float32
    )
    timestamp_input: tf.Tensor = keras.layers.Input(
        shape=(), name="scheduled_timestamp", dtype=tf.float32
    )

    # Embeddings for categorical features
    station_emb: tf.Tensor = keras.layers.Embedding(vocab_info["num_stations"] + 1, 16)(
        station_input
    )
    station_emb = keras.layers.Flatten()(station_emb)

    route_emb: tf.Tensor = keras.layers.Embedding(vocab_info["num_routes"] + 1, 8)(
        route_input
    )
    route_emb = keras.layers.Flatten()(route_emb)

    direction_emb: tf.Tensor = keras.layers.Embedding(3, 4)(
        direction_input
    )  # 0, 1, or 2
    direction_emb = keras.layers.Flatten()(direction_emb)

    # Normalize timestamp
    timestamp_norm: tf.Tensor = keras.layers.Normalization(axis=None)(timestamp_input)

    # Expand scalar features to match embedding dimensions
    hour_sin_expanded = keras.layers.Reshape((1,))(hour_sin_input)
    hour_cos_expanded = keras.layers.Reshape((1,))(hour_cos_input)
    minute_sin_expanded = keras.layers.Reshape((1,))(minute_sin_input)
    minute_cos_expanded = keras.layers.Reshape((1,))(minute_cos_input)
    day_sin_expanded = keras.layers.Reshape((1,))(day_sin_input)
    day_cos_expanded = keras.layers.Reshape((1,))(day_cos_input)
    timestamp_expanded = keras.layers.Reshape((1,))(timestamp_norm)

    # Concatenate all features
    concat_features: tf.Tensor = keras.layers.Concatenate()(
        [
            station_emb,
            route_emb,
            direction_emb,
            hour_sin_expanded,
            hour_cos_expanded,
            minute_sin_expanded,
            minute_cos_expanded,
            day_sin_expanded,
            day_cos_expanded,
            timestamp_expanded,
        ]
    )

    # Dense layers
    x: tf.Tensor = keras.layers.Dense(128, activation="relu")(concat_features)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)

    # Output layer
    outputs: tf.Tensor = keras.layers.Dense(
        vocab_info["num_tracks"], activation="softmax"
    )(x)

    # Create model
    model = keras.Model(
        inputs=[
            station_input,
            route_input,
            direction_input,
            hour_sin_input,
            hour_cos_input,
            minute_sin_input,
            minute_cos_input,
            day_sin_input,
            day_cos_input,
            timestamp_input,
        ],
        outputs=outputs,
    )

    return model


def build_tunable_model(
    hp: kt.HyperParameters, vocab_info: dict[str, Any]
) -> keras.Model:
    """Create a tunable neural network for track prediction with hyperparameter optimization."""

    # Input layers
    station_input: tf.Tensor = keras.layers.Input(
        shape=(), name="station_id", dtype=tf.int64
    )
    route_input: tf.Tensor = keras.layers.Input(
        shape=(), name="route_id", dtype=tf.int64
    )
    direction_input: tf.Tensor = keras.layers.Input(
        shape=(), name="direction_id", dtype=tf.int64
    )

    # Time features
    hour_sin_input: tf.Tensor = keras.layers.Input(
        shape=(), name="hour_sin", dtype=tf.float32
    )
    hour_cos_input: tf.Tensor = keras.layers.Input(
        shape=(), name="hour_cos", dtype=tf.float32
    )
    minute_sin_input: tf.Tensor = keras.layers.Input(
        shape=(), name="minute_sin", dtype=tf.float32
    )
    minute_cos_input: tf.Tensor = keras.layers.Input(
        shape=(), name="minute_cos", dtype=tf.float32
    )
    day_sin_input: tf.Tensor = keras.layers.Input(
        shape=(), name="day_sin", dtype=tf.float32
    )
    day_cos_input: tf.Tensor = keras.layers.Input(
        shape=(), name="day_cos", dtype=tf.float32
    )
    timestamp_input: tf.Tensor = keras.layers.Input(
        shape=(), name="scheduled_timestamp", dtype=tf.float32
    )

    # Tunable embedding dimensions
    station_emb_dim = hp.Int("station_embedding_dim", min_value=8, max_value=32, step=8)
    route_emb_dim = hp.Int("route_embedding_dim", min_value=4, max_value=16, step=4)
    direction_emb_dim = hp.Int(
        "direction_embedding_dim", min_value=2, max_value=8, step=2
    )

    # Embeddings for categorical features
    station_emb: tf.Tensor = keras.layers.Embedding(
        vocab_info["num_stations"] + 1, station_emb_dim
    )(station_input)
    station_emb = keras.layers.Flatten()(station_emb)

    route_emb: tf.Tensor = keras.layers.Embedding(
        vocab_info["num_routes"] + 1, route_emb_dim
    )(route_input)
    route_emb = keras.layers.Flatten()(route_emb)

    direction_emb: tf.Tensor = keras.layers.Embedding(3, direction_emb_dim)(
        direction_input
    )
    direction_emb = keras.layers.Flatten()(direction_emb)

    # Normalize timestamp
    timestamp_norm: tf.Tensor = keras.layers.Normalization(axis=None)(timestamp_input)

    # Expand scalar features to match embedding dimensions
    hour_sin_expanded = keras.layers.Reshape((1,))(hour_sin_input)
    hour_cos_expanded = keras.layers.Reshape((1,))(hour_cos_input)
    minute_sin_expanded = keras.layers.Reshape((1,))(minute_sin_input)
    minute_cos_expanded = keras.layers.Reshape((1,))(minute_cos_input)
    day_sin_expanded = keras.layers.Reshape((1,))(day_sin_input)
    day_cos_expanded = keras.layers.Reshape((1,))(day_cos_input)
    timestamp_expanded = keras.layers.Reshape((1,))(timestamp_norm)

    # Concatenate all features
    concat_features: tf.Tensor = keras.layers.Concatenate()(
        [
            station_emb,
            route_emb,
            direction_emb,
            hour_sin_expanded,
            hour_cos_expanded,
            minute_sin_expanded,
            minute_cos_expanded,
            day_sin_expanded,
            day_cos_expanded,
            timestamp_expanded,
        ]
    )

    # Tunable dense layers
    x = concat_features
    num_layers = hp.Int("num_layers", min_value=2, max_value=4)
    for i in range(num_layers):
        units = hp.Int(f"units_{i}", min_value=32, max_value=256, step=32)
        x = keras.layers.Dense(units, activation="relu")(x)

        dropout_rate = hp.Float(f"dropout_{i}", min_value=0.1, max_value=0.5, step=0.1)
        x = keras.layers.Dropout(dropout_rate)(x)

    # Output layer
    outputs: tf.Tensor = keras.layers.Dense(
        vocab_info["num_tracks"], activation="softmax"
    )(x)

    # Create model
    model = keras.Model(
        inputs=[
            station_input,
            route_input,
            direction_input,
            hour_sin_input,
            hour_cos_input,
            minute_sin_input,
            minute_cos_input,
            day_sin_input,
            day_cos_input,
            timestamp_input,
        ],
        outputs=outputs,
    )

    # Tunable learning rate
    learning_rate = hp.Float(
        "learning_rate", min_value=1e-4, max_value=1e-2, sampling="log"
    )
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def tune_hyperparameters(
    data_dir: str,
    project_name: str = "track_prediction_tuning",
    max_trials: int = 20,
    epochs_per_trial: int = 10,
    batch_size: int = 32,
) -> kt.Tuner:
    """Tune hyperparameters using Keras Tuner."""

    # Load dataset
    print(f"Loading dataset from {data_dir}...")
    train_ds, val_ds, vocab_info = load_tfrecord_dataset(
        data_dir, batch_size=batch_size
    )

    print("Vocabulary sizes:")
    print(f"  Stations: {vocab_info['num_stations']}")
    print(f"  Routes: {vocab_info['num_routes']}")
    print(f"  Tracks: {vocab_info['num_tracks']}")

    # Create tuner
    tuner = kt.RandomSearch(
        hypermodel=lambda hp: build_tunable_model(hp, vocab_info),
        objective="val_accuracy",
        max_trials=max_trials,
        project_name=project_name,
        overwrite=True,
    )

    print(f"Starting hyperparameter tuning with {max_trials} trials...")
    print("Search space:")
    tuner.search_space_summary()

    # Setup callbacks for tuning
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=0
        ),
    ]

    # Search for best hyperparameters
    tuner.search(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_per_trial,
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
) -> keras.Model:
    """Train the best model found by hyperparameter tuning."""

    # Load dataset
    print(f"Loading dataset from {data_dir}...")
    train_ds, val_ds, vocab_info = load_tfrecord_dataset(
        data_dir, batch_size=batch_size
    )

    # Get best hyperparameters and build model
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = build_tunable_model(best_hps, vocab_info)

    print("Training best model with hyperparameters:")
    for param in best_hps.space:
        print(f"  {param.name}: {best_hps.get(param.name)}")

    print(model.summary())

    # Setup callbacks
    # callbacks = [
    #     keras.callbacks.EarlyStopping(
    #         monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
    #     ),
    #     keras.callbacks.ReduceLROnPlateau(
    #         monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
    #     ),
    #     keras.callbacks.ModelCheckpoint(
    #         filepath=f"{model_save_path}_best.keras",
    #         monitor="val_accuracy",
    #         save_best_only=True,
    #         verbose=1,
    #     ),
    # ]

    # Train model
    print(f"Training model for {epochs} epochs...")
    # history = model.fit(
    #     train_ds,
    #     validation_data=val_ds,
    #     epochs=epochs,
    #     steps_per_epoch=vocab_info["train_steps_per_epoch"],
    #     validation_steps=vocab_info["val_steps_per_epoch"],
    #     callbacks=callbacks,
    #     verbose=1,
    # )

    # Save final model
    model.save(f"{model_save_path}_final.keras")
    print(f"Model saved to {model_save_path}_final.keras")

    # Save hyperparameters and vocabulary info
    save_path = f"{model_save_path}_config.json"
    config = {
        "hyperparameters": {
            param.name: best_hps.get(param.name) for param in best_hps.space
        },
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

    return model


def train_model(
    data_dir: str,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    model_save_path: str = "track_prediction_model",
) -> Any:
    """Train the track prediction model with proper callbacks and saving."""

    # Load dataset
    print(f"Loading dataset from {data_dir}...")
    train_ds, val_ds, vocab_info = load_tfrecord_dataset(
        data_dir, batch_size=batch_size
    )

    print("Vocabulary sizes:")
    print(f"  Stations: {vocab_info['num_stations']}")
    print(f"  Routes: {vocab_info['num_routes']}")
    print(f"  Tracks: {vocab_info['num_tracks']}")

    # Create and compile model
    print("Creating model...")
    model = create_simple_model(vocab_info)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(model.summary())

    # Setup callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
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
    ]

    # Train model
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

    return history


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Train track prediction model")
    parser.add_argument("data_dir", help="Path to exported TFRecord data")

    # Add subcommands for different training modes
    subparsers = parser.add_subparsers(dest="command", help="Training mode")

    # Standard training command
    train_parser = subparsers.add_parser(
        "train", help="Train model with fixed hyperparameters"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    train_parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate"
    )
    train_parser.add_argument(
        "--model-path", default="track_prediction_model", help="Model save path prefix"
    )

    # Hyperparameter tuning command
    tune_parser = subparsers.add_parser(
        "tune", help="Tune hyperparameters and train best model"
    )
    tune_parser.add_argument(
        "--max-trials", type=int, default=20, help="Maximum number of tuning trials"
    )
    tune_parser.add_argument(
        "--epochs-per-trial",
        type=int,
        default=10,
        help="Epochs per trial during tuning",
    )
    tune_parser.add_argument(
        "--final-epochs", type=int, default=50, help="Epochs for final training"
    )
    tune_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    tune_parser.add_argument(
        "--project-name",
        default="track_prediction_tuning",
        help="Keras Tuner project name",
    )
    tune_parser.add_argument(
        "--model-path",
        default="track_prediction_model_tuned",
        help="Model save path prefix",
    )

    # Set default command to train for backward compatibility
    args = parser.parse_args()
    if args.command is None:
        # If no subcommand provided, default to training with old behavior
        args.command = "train"
        args.epochs = 50
        args.batch_size = 32
        args.learning_rate = 0.001
        args.model_path = "track_prediction_model"

    try:
        if args.command == "train":
            train_model(
                data_dir=args.data_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                model_save_path=args.model_path,
            )
        elif args.command == "tune":
            # First tune hyperparameters
            tuner = tune_hyperparameters(
                data_dir=args.data_dir,
                project_name=args.project_name,
                max_trials=args.max_trials,
                epochs_per_trial=args.epochs_per_trial,
                batch_size=args.batch_size,
            )

            # Then train the best model
            train_best_model(
                tuner=tuner,
                data_dir=args.data_dir,
                epochs=args.final_epochs,
                batch_size=args.batch_size,
                model_save_path=args.model_path,
            )
    except Exception as e:
        print(f"Error during {args.command}: {e}")
        sys.exit(1)
