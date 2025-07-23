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

    # Batch and prefetch
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Create vocab info for model building
    vocab_info = {
        "station_vocab": station_vocab,
        "route_vocab": route_vocab,
        "track_vocab": track_vocab,
        "num_stations": len(station_vocab),
        "num_routes": len(route_vocab),
        "num_tracks": len(track_vocab),
        "metadata": metadata,
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
    timestamp_norm: tf.Tensor = keras.layers.Normalization()(timestamp_input)

    # Concatenate all features
    concat_features: tf.Tensor = keras.layers.Concatenate()(
        [
            station_emb,
            route_emb,
            direction_emb,
            hour_sin_input,
            hour_cos_input,
            minute_sin_input,
            minute_cos_input,
            day_sin_input,
            day_cos_input,
            timestamp_norm,
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


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) != 2:
        print("Usage: python tfrecord_reader.py <path_to_exported_data>")
        sys.exit(1)

    data_dir = sys.argv[1]

    # Load dataset
    train_ds, val_ds, vocab_info = load_tfrecord_dataset(data_dir)

    print(f"Train dataset: {train_ds}")
    print(f"Validation dataset: {val_ds}")
    print(f"Vocabulary info: {vocab_info}")

    # Create and compile model
    model = create_simple_model(vocab_info)
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    print(model.summary())

    # Train for a few epochs as demo
    print("Training model for 3 epochs...")
    model.fit(train_ds, validation_data=val_ds, epochs=3)
