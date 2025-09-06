"""
Model architecture definitions for track prediction.

This module contains functions to create different model architectures
including simple models and tunable models for hyperparameter optimization.
"""

from typing import Any

import keras
import tensorflow as tf


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
    inputs: dict[str, tf.Tensor],
    vocab_info: dict[str, Any],
    use_regularization: bool = True,
) -> dict[str, tf.Tensor]:
    """Create embedding layers for categorical features with optional regularization."""
    regularizer = keras.regularizers.l2(1e-4) if use_regularization else None

    station_emb = keras.layers.Flatten()(
        keras.layers.Embedding(
            vocab_info["num_stations"] + 1, 16, embeddings_regularizer=regularizer
        )(inputs["station_id"])
    )
    route_emb = keras.layers.Flatten()(
        keras.layers.Embedding(
            vocab_info["num_routes"] + 1, 8, embeddings_regularizer=regularizer
        )(inputs["route_id"])
    )
    direction_emb = keras.layers.Flatten()(
        keras.layers.Embedding(3, 4, embeddings_regularizer=regularizer)(
            inputs["direction_id"]
        )
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


def create_simple_model(
    vocab_info: dict[str, Any], use_regularization: bool = True
) -> keras.Model:
    """Create a simple neural network for track prediction using functional API with regularization."""

    # Create input layers
    inputs = _create_input_layers()

    # Create embeddings with regularization
    embeddings = _create_embedding_layers(inputs, vocab_info, use_regularization)

    # Create time features
    time_features = _create_time_features(inputs)

    # Concatenate all features
    all_features = list(embeddings.values()) + time_features
    concat_features = keras.layers.Concatenate()(all_features)

    # Add batch normalization for feature stability
    if use_regularization:
        concat_features = keras.layers.BatchNormalization()(concat_features)

    # Build dense layers with regularization
    x = keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)
        if use_regularization
        else None,
    )(concat_features)
    if use_regularization:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.4)(x)  # Increased dropout for small datasets

    x = keras.layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)
        if use_regularization
        else None,
    )(x)
    if use_regularization:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.4)(x)

    # Output layer
    outputs = keras.layers.Dense(vocab_info["num_tracks"], activation="softmax")(x)

    # Create model
    model = keras.Model(inputs=list(inputs.values()), outputs=outputs)

    return model


def _create_tunable_embedding_layers(
    inputs: dict[str, tf.Tensor], vocab_info: dict[str, Any], hp: Any
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


def _create_tunable_dense_layers(x: tf.Tensor, hp: Any) -> tf.Tensor:
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


def build_tunable_model(hp: Any, vocab_info: dict[str, Any]) -> keras.Model:
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


def build_model_from_config(
    config: dict[str, Any], vocab_info: dict[str, Any]
) -> keras.Model:
    """Build a model using a plain config dict (for Ray Tune).

    The config is expected to contain keys similar to the Keras Tuner space:
      - station_embedding_dim, route_embedding_dim, direction_embedding_dim
      - num_layers, units_0..units_{n-1}, dropout_0..dropout_{n-1}
      - learning_rate
    """
    # Inputs and time features
    inputs = _create_input_layers()

    # Embeddings
    station_dim = int(config.get("station_embedding_dim", 16))
    route_dim = int(config.get("route_embedding_dim", 8))
    direction_dim = int(config.get("direction_embedding_dim", 4))

    station_emb = keras.layers.Flatten()(
        keras.layers.Embedding(vocab_info["num_stations"] + 1, station_dim)(
            inputs["station_id"]
        )
    )
    route_emb = keras.layers.Flatten()(
        keras.layers.Embedding(vocab_info["num_routes"] + 1, route_dim)(
            inputs["route_id"]
        )
    )
    direction_emb = keras.layers.Flatten()(
        keras.layers.Embedding(3, direction_dim)(inputs["direction_id"])
    )

    time_features = _create_time_features(inputs)
    all_features = [station_emb, route_emb, direction_emb] + time_features
    concat_features = keras.layers.Concatenate()(all_features)

    # Dense stack
    x = concat_features
    num_layers = int(config.get("num_layers", 3))
    for i in range(num_layers):
        units = int(config.get(f"units_{i}", 64))
        dropout = float(config.get(f"dropout_{i}", 0.3))
        x = keras.layers.Dense(units, activation="relu")(x)
        x = keras.layers.Dropout(dropout)(x)

    outputs = keras.layers.Dense(vocab_info["num_tracks"], activation="softmax")(x)
    model = keras.Model(inputs=list(inputs.values()), outputs=outputs)

    lr = float(config.get("learning_rate", 1e-3))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
