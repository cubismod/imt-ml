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
from tqdm import tqdm


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

    # Add progress bar for file loading
    with tqdm(
        total=len(tfrecord_files), desc="Loading TFRecord files", unit="files"
    ) as pbar:
        pbar.update(len(tfrecord_files))

    # Create dataset with interleaved parallel file reading
    dataset = tf.data.Dataset.from_tensor_slices([str(f) for f in tfrecord_files])
    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=min(len(tfrecord_files), tf.data.AUTOTUNE),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,  # Allow reordering for better performance
    )
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # Build vocabularies by scanning the dataset in parallel
    print("Building vocabularies...")
    stations = set()
    routes = set()
    tracks = set()

    # Use parallel map to collect vocabulary items
    vocab_dataset = dataset.map(
        lambda x: {
            "station": x["station_id"],
            "route": x["route_id"],
            "track": x["track_number"],
        },
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Get total count for progress bar
    total_records = metadata.get("total_records", 0)

    for example in tqdm(
        vocab_dataset.take(-1),
        desc="Building vocabularies",
        total=total_records,
        unit="records",
    ):  # Scan all examples
        stations.add(example["station"].numpy().decode("utf-8"))
        routes.add(example["route"].numpy().decode("utf-8"))
        track_str = example["track"].numpy().decode("utf-8")
        if track_str:  # Only add non-empty tracks
            tracks.add(track_str)

    station_vocab = sorted(list(stations))
    route_vocab = sorted(list(routes))
    track_vocab = sorted(list(tracks))

    print(
        f"Vocabularies: {len(station_vocab)} stations, {len(route_vocab)} routes, {len(track_vocab)} tracks"
    )

    print("Applying feature engineering...")
    with tqdm(desc="Processing features", unit="batch") as pbar:
        pass  # The actual processing happens in the TensorFlow pipeline

    feature_fn = create_feature_engineering_fn(station_vocab, route_vocab, track_vocab)
    dataset = dataset.map(feature_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.filter(lambda x, y: y < len(track_vocab))

    total_size = metadata["total_records"]
    train_size = int(total_size * train_split)

    dataset = dataset.shuffle(
        min(shuffle_buffer, total_size), reshuffle_each_iteration=False
    )

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)

    train_dataset = (
        train_dataset.repeat()
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_dataset = (
        val_dataset.repeat()
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

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


def _create_input_layers() -> dict[str, tf.Tensor]:
    """Create all input layers for the model."""
    return {
        "station_id": keras.layers.Input(shape=(), name="station_id", dtype=tf.int64),
        "route_id": keras.layers.Input(shape=(), name="route_id", dtype=tf.int64),
        "direction_id": keras.layers.Input(
            shape=(), name="direction_id", dtype=tf.int64
        ),
        "hour_sin": keras.layers.Input(shape=(), name="hour_sin", dtype=tf.float32),
        "hour_cos": keras.layers.Input(shape=(), name="hour_cos", dtype=tf.float32),
        "minute_sin": keras.layers.Input(shape=(), name="minute_sin", dtype=tf.float32),
        "minute_cos": keras.layers.Input(shape=(), name="minute_cos", dtype=tf.float32),
        "day_sin": keras.layers.Input(shape=(), name="day_sin", dtype=tf.float32),
        "day_cos": keras.layers.Input(shape=(), name="day_cos", dtype=tf.float32),
        "scheduled_timestamp": keras.layers.Input(
            shape=(), name="scheduled_timestamp", dtype=tf.float32
        ),
    }


def _create_embedding_layers(
    inputs: dict[str, tf.Tensor], vocab_info: dict[str, Any]
) -> dict[str, tf.Tensor]:
    """Create embedding layers for categorical features."""
    station_emb = keras.layers.Flatten()(
        keras.layers.Embedding(vocab_info["num_stations"] + 1, 16)(inputs["station_id"])
    )
    route_emb = keras.layers.Flatten()(
        keras.layers.Embedding(vocab_info["num_routes"] + 1, 8)(inputs["route_id"])
    )
    direction_emb = keras.layers.Flatten()(
        keras.layers.Embedding(3, 4)(inputs["direction_id"])
    )

    return {
        "station_emb": station_emb,
        "route_emb": route_emb,
        "direction_emb": direction_emb,
    }


def _create_time_features(inputs: dict[str, tf.Tensor]) -> list[tf.Tensor]:
    """Create normalized time features."""
    time_features = []
    for feature_name in [
        "hour_sin",
        "hour_cos",
        "minute_sin",
        "minute_cos",
        "day_sin",
        "day_cos",
    ]:
        expanded = keras.layers.Reshape((1,))(inputs[feature_name])
        time_features.append(expanded)

    # Normalize and expand timestamp
    timestamp_norm = keras.layers.Normalization(axis=None)(
        inputs["scheduled_timestamp"]
    )
    timestamp_expanded = keras.layers.Reshape((1,))(timestamp_norm)
    time_features.append(timestamp_expanded)

    return time_features


def create_simple_model(vocab_info: dict[str, Any]) -> keras.Model:
    """Create a simple neural network for track prediction using functional API."""

    # Create input layers
    inputs = _create_input_layers()

    # Create embeddings
    embeddings = _create_embedding_layers(inputs, vocab_info)

    # Create time features
    time_features = _create_time_features(inputs)

    # Concatenate all features
    all_features = list(embeddings.values()) + time_features
    concat_features = keras.layers.Concatenate()(all_features)

    # Build dense layers functionally
    x = keras.layers.Dense(128, activation="relu")(concat_features)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)

    # Output layer
    outputs = keras.layers.Dense(vocab_info["num_tracks"], activation="softmax")(x)

    # Create model
    model = keras.Model(inputs=list(inputs.values()), outputs=outputs)

    return model


def _create_tunable_embedding_layers(
    inputs: dict[str, tf.Tensor], vocab_info: dict[str, Any], hp: kt.HyperParameters
) -> dict[str, tf.Tensor]:
    """Create tunable embedding layers for categorical features."""
    # Tunable embedding dimensions
    station_emb_dim = hp.Int("station_embedding_dim", min_value=8, max_value=64, step=8)
    route_emb_dim = hp.Int("route_embedding_dim", min_value=4, max_value=64, step=4)
    direction_emb_dim = hp.Int(
        "direction_embedding_dim", min_value=2, max_value=32, step=2
    )

    station_emb = keras.layers.Flatten()(
        keras.layers.Embedding(vocab_info["num_stations"] + 1, station_emb_dim)(
            inputs["station_id"]
        )
    )
    route_emb = keras.layers.Flatten()(
        keras.layers.Embedding(vocab_info["num_routes"] + 1, route_emb_dim)(
            inputs["route_id"]
        )
    )
    direction_emb = keras.layers.Flatten()(
        keras.layers.Embedding(3, direction_emb_dim)(inputs["direction_id"])
    )

    return {
        "station_emb": station_emb,
        "route_emb": route_emb,
        "direction_emb": direction_emb,
    }


def _create_tunable_dense_layers(x: tf.Tensor, hp: kt.HyperParameters) -> tf.Tensor:
    """Create tunable dense layers with hyperparameter optimization."""
    num_layers = hp.Int("num_layers", min_value=2, max_value=15)

    for i in range(num_layers):
        units = hp.Int(f"units_{i}", min_value=8, max_value=256, step=4)
        x = keras.layers.Dense(units, activation="relu")(x)

        dropout_rate = hp.Float(
            f"dropout_{i}", min_value=0.01, max_value=0.6, step=0.01
        )
        x = keras.layers.Dropout(dropout_rate)(x)

    return x


def build_tunable_model(
    hp: kt.HyperParameters, vocab_info: dict[str, Any]
) -> keras.Model:
    """Create a tunable neural network for track prediction with hyperparameter optimization."""

    # Create input layers
    inputs = _create_input_layers()

    # Create tunable embeddings
    embeddings = _create_tunable_embedding_layers(inputs, vocab_info, hp)

    # Create time features
    time_features = _create_time_features(inputs)

    # Concatenate all features
    all_features = list(embeddings.values()) + time_features
    concat_features = keras.layers.Concatenate()(all_features)

    # Build tunable dense layers
    x = _create_tunable_dense_layers(concat_features, hp)

    # Output layer
    outputs = keras.layers.Dense(vocab_info["num_tracks"], activation="softmax")(x)

    # Create model
    model = keras.Model(inputs=list(inputs.values()), outputs=outputs)

    # Tunable learning rate and compilation
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
    model.fit(
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
        "tune",
        help="Tune hyperparameters using Hyperband algorithm and train best model",
    )
    tune_parser.add_argument(
        "--max-epochs",
        type=int,
        default=50,
        help="Maximum epochs for Hyperband algorithm",
    )
    tune_parser.add_argument(
        "--factor",
        type=int,
        default=3,
        help="Hyperband reduction factor",
    )
    tune_parser.add_argument(
        "--hyperband-iterations",
        type=int,
        default=1,
        help="Number of Hyperband iterations",
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
    tune_parser.add_argument(
        "--distributed", action="store_true", help="Use distributed training"
    )
    tune_parser.add_argument(
        "--executions-per-trial",
        type=int,
        default=2,
        help="Number of executions per trial",
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
                max_epochs=args.max_epochs,
                factor=args.factor,
                hyperband_iterations=args.hyperband_iterations,
                batch_size=args.batch_size,
                distributed=args.distributed,
                executions_per_trial=args.executions_per_trial,
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
