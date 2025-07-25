"""
Data processing and TFRecord loading utilities for track prediction models.

This module handles loading TFRecord files, building vocabularies, applying
feature engineering, and creating training/validation datasets.
"""

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
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


def _apply_data_augmentation(
    features: dict[str, tf.Tensor], target: tf.Tensor
) -> tuple[dict[str, tf.Tensor], tf.Tensor]:
    """Apply data augmentation for small datasets."""
    # Add noise to continuous features to create variations
    noise_factor = 0.05

    # Add small noise to timestamp (within 5 minutes)
    timestamp_noise = tf.random.normal([], stddev=300.0)  # 5 minutes in seconds
    features["scheduled_timestamp"] = features["scheduled_timestamp"] + timestamp_noise

    # Add small noise to time features
    for time_feature in [
        "hour_sin",
        "hour_cos",
        "minute_sin",
        "minute_cos",
        "day_sin",
        "day_cos",
    ]:
        noise = tf.random.normal([], stddev=noise_factor)
        features[time_feature] = tf.clip_by_value(
            features[time_feature] + noise, -1.0, 1.0
        )

    return features, target


def _calculate_class_weights(
    dataset: tf.data.Dataset, num_classes: int
) -> dict[int, float]:
    """Calculate class weights for imbalanced dataset."""
    # Collect all labels
    labels = []
    for _, target in dataset.take(-1):
        labels.append(target.numpy())

    labels = np.array(labels)  # type: ignore[assignment]

    # Calculate class weights
    class_weights = compute_class_weight(
        "balanced", classes=np.arange(num_classes), y=labels
    )

    return {i: weight for i, weight in enumerate(class_weights)}


def load_tfrecord_dataset(
    data_dir: str,
    batch_size: int = 32,
    train_split: float = 0.8,
    shuffle_buffer: int = 10000,
    augment_data: bool = True,
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

    # Calculate class weights for imbalanced datasets
    print("Calculating class weights...")
    class_weights = _calculate_class_weights(dataset, len(track_vocab))

    total_size = metadata["total_records"]
    train_size = int(total_size * train_split)

    dataset = dataset.shuffle(
        min(shuffle_buffer, total_size), reshuffle_each_iteration=False
    )

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)

    # Apply data augmentation to training set for small datasets
    if augment_data and total_size < 10000:
        print("Applying data augmentation for small dataset...")
        augmented_train = train_dataset.map(
            _apply_data_augmentation, num_parallel_calls=tf.data.AUTOTUNE
        )
        # Combine original and augmented data
        train_dataset = train_dataset.concatenate(augmented_train)

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
        "class_weights": class_weights,
    }

    return train_dataset, val_dataset, vocab_info
