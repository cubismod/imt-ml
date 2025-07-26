"""
TensorFlow helper functions for reading and preprocessing the exported track data.

This module provides utilities to read TFRecord files and create TensorFlow datasets
for training track prediction models.

This is the main entry point that combines functionality from specialized modules:
- dataset: Data processing and TFRecord loading
- models: Model architecture definitions
- training: Training utilities and functions
- reporting: Training reports and output management
"""

import sys
import time

import click

# Import main functions to maintain backward compatibility
from imt_ml.dataset import (
    create_feature_engineering_fn,
    load_tfrecord_dataset,
    parse_tfrecord_fn,
)
from imt_ml.models import (
    build_tunable_model,
    create_simple_model,
)
from imt_ml.reporting import (
    create_timestamped_output_dir,
    generate_training_report,
)
from imt_ml.training import (
    evaluate_with_cross_validation,
    train_best_model,
    train_ensemble_model,
    train_model,
    tune_hyperparameters,
)


# Command Line Interface
@click.group(invoke_without_command=True)
@click.argument("data_dir", type=click.Path(exists=True))
@click.pass_context
def cli(ctx, data_dir):
    """Train track prediction model."""
    ctx.ensure_object(dict)
    ctx.obj["data_dir"] = data_dir

    # Default to train command if no subcommand provided
    if ctx.invoked_subcommand is None:
        ctx.invoke(
            train,
            epochs=50,
            batch_size=32,
            learning_rate=0.001,
            model_path="track_prediction_model",
        )


@cli.command()
@click.option("--epochs", default=50, help="Number of training epochs")
@click.option("--batch-size", default=32, help="Batch size")
@click.option("--learning-rate", default=0.001, help="Learning rate")
@click.option(
    "--model-path", default="track_prediction_model", help="Model save path prefix"
)
@click.pass_context
def train(ctx, epochs, batch_size, learning_rate, model_path):
    """Train model with fixed hyperparameters."""
    try:
        full_model_path = create_timestamped_output_dir("train", model_path)
        train_model(
            data_dir=ctx.obj["data_dir"],
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            model_save_path=full_model_path,
            generate_report_func=generate_training_report,
        )
    except Exception as e:
        click.echo(f"Error during training: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--num-models", default=3, help="Number of models in ensemble")
@click.option("--epochs", default=50, help="Number of training epochs")
@click.option("--batch-size", default=32, help="Batch size")
@click.option("--learning-rate", default=0.001, help="Learning rate")
@click.option(
    "--model-path",
    default="track_prediction_ensemble",
    help="Model save path prefix",
)
@click.pass_context
def ensemble(ctx, num_models, epochs, batch_size, learning_rate, model_path):
    """Train ensemble of models for better accuracy."""
    try:
        full_model_path = create_timestamped_output_dir("ensemble", model_path)
        train_ensemble_model(
            data_dir=ctx.obj["data_dir"],
            num_models=num_models,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            model_save_path=full_model_path,
            generate_report_func=generate_training_report,
        )
    except Exception as e:
        click.echo(f"Error during ensemble training: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--k-folds", default=5, help="Number of folds for cross-validation")
@click.option("--epochs", default=30, help="Number of training epochs per fold")
@click.option("--batch-size", default=32, help="Batch size")
@click.pass_context
def cv(ctx, k_folds, epochs, batch_size):
    """Evaluate model using cross-validation."""
    try:
        full_model_path = create_timestamped_output_dir("cv", "cv_evaluation")
        evaluate_with_cross_validation(
            data_dir=ctx.obj["data_dir"],
            k_folds=k_folds,
            epochs=epochs,
            batch_size=batch_size,
            model_save_path=full_model_path,
            generate_report_func=generate_training_report,
        )
    except Exception as e:
        click.echo(f"Error during cross-validation: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--max-epochs", default=50, help="Maximum epochs for Hyperband algorithm")
@click.option("--factor", default=3, help="Hyperband reduction factor")
@click.option(
    "--hyperband-iterations", default=1, help="Number of Hyperband iterations"
)
@click.option("--final-epochs", default=50, help="Epochs for final training")
@click.option("--batch-size", default=32, help="Batch size")
@click.option(
    "--project-name",
    default="track_prediction_tuning",
    help="Keras Tuner project name",
)
@click.option(
    "--model-path",
    default="track_prediction_model_tuned",
    help="Model save path prefix",
)
@click.option("--distributed", is_flag=True, help="Use distributed training")
@click.option(
    "--executions-per-trial", default=2, help="Number of executions per trial"
)
@click.pass_context
def tune(
    ctx,
    max_epochs,
    factor,
    hyperband_iterations,
    final_epochs,
    batch_size,
    project_name,
    model_path,
    distributed,
    executions_per_trial,
):
    """Tune hyperparameters using Hyperband algorithm and train best model."""
    try:
        full_model_path = create_timestamped_output_dir("tune", model_path)
        # First tune hyperparameters
        tuning_start_time = time.time()
        tuner = tune_hyperparameters(
            data_dir=ctx.obj["data_dir"],
            project_name=project_name,
            max_epochs=max_epochs,
            factor=factor,
            hyperband_iterations=hyperband_iterations,
            batch_size=batch_size,
            distributed=distributed,
            executions_per_trial=executions_per_trial,
        )
        tuning_time = time.time() - tuning_start_time

        # Then train the best model
        train_best_model(
            tuner=tuner,
            data_dir=ctx.obj["data_dir"],
            epochs=final_epochs,
            batch_size=batch_size,
            model_save_path=full_model_path,
            tuning_time=tuning_time,
            generate_report_func=generate_training_report,
        )
    except Exception as e:
        click.echo(f"Error during hyperparameter tuning: {e}", err=True)
        sys.exit(1)


# Export main functions for external use
__all__ = [
    # Dataset functions
    "load_tfrecord_dataset",
    "parse_tfrecord_fn",
    "create_feature_engineering_fn",
    # Model functions
    "create_simple_model",
    "build_tunable_model",
    # Training functions
    "train_model",
    "train_ensemble_model",
    "tune_hyperparameters",
    "train_best_model",
    "evaluate_with_cross_validation",
    # Utility functions
    "create_timestamped_output_dir",
    "generate_training_report",
    # CLI
    "cli",
]


if __name__ == "__main__":
    cli()
