"""
CLI and utility to export a Keras `.keras` model to a TensorFlow Lite `.tflite` file.

Supports optional post-training quantization modes that do not require a
representative dataset: dynamic range and float16.
"""

from __future__ import annotations

import os
from typing import Literal

import click
import tensorflow as tf
from keras import models as keras_models

QuantizationMode = Literal["none", "dynamic", "float16"]


def convert_keras_to_tflite(
    input_model_path: str,
    output_tflite_path: str | None = None,
    *,
    quantization: QuantizationMode = "none",
) -> str:
    """Convert a `.keras` model file to a `.tflite` file.

    Parameters
    - input_model_path: Path to the saved `.keras` model file.
    - output_tflite_path: Desired `.tflite` output path. If not provided, it will
      be derived from the input path by replacing the extension with `.tflite`.
    - quantization: Optional post-training quantization mode.
      - "none": No quantization (float32).
      - "dynamic": Dynamic range quantization (weights quantized; activations float32).
      - "float16": Weight quantization to float16 (compatible with many mobile GPUs/NPUs).

    Returns
    - The path to the written `.tflite` file.
    """
    if not os.path.exists(input_model_path):
        raise FileNotFoundError(f"Model not found: {input_model_path}")

    if output_tflite_path is None:
        base, _ = os.path.splitext(input_model_path)
        output_tflite_path = f"{base}.tflite"

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(output_tflite_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Load model and configure converter
    model = keras_models.load_model(input_model_path, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantization == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantization == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    with open(output_tflite_path, "wb") as f:
        f.write(tflite_model)

    return output_tflite_path


@click.command()
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False),
    default=None,
    help="Output .tflite path (defaults to MODEL_PATH with .tflite extension)",
)
@click.option(
    "--quantization",
    type=click.Choice(["none", "dynamic", "float16"], case_sensitive=False),
    default="none",
    show_default=True,
    help=(
        "Post-training quantization: none (float32), dynamic (no dataset), or float16"
    ),
)
def cli(model_path: str, output_path: str | None, quantization: str) -> None:
    """Export a `.keras` model to `.tflite`.

    Example:
    uv run tflite-export ./output/train_.../track_prediction_model_final.keras \
        --quantization float16 --output ./output/model_float16.tflite
    """
    qt: QuantizationMode = quantization.lower()  # type: ignore[assignment]
    out_path = convert_keras_to_tflite(model_path, output_path, quantization=qt)
    size_kb = os.path.getsize(out_path) / 1024.0
    click.echo(f"TFLite model written to {out_path} ({size_kb:.1f} KB)")


__all__ = ["convert_keras_to_tflite", "cli"]


if __name__ == "__main__":
    cli()
