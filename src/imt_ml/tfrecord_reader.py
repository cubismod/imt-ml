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
    train_best_model_from_config,
    train_ensemble_model,
    train_model,
    tune_hyperparameters_ray,
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
    "--early-stop-patience",
    type=int,
    default=None,
    help="Early stopping patience (disable by setting 0 or use --no-early-stop)",
)
@click.option(
    "--early-stop-min-delta",
    type=float,
    default=1e-4,
    help="Minimum delta for early stopping and LR plateau",
)
@click.option(
    "--reduce-lr-patience",
    type=int,
    default=None,
    help="Patience for ReduceLROnPlateau (defaults relative to early stop)",
)
@click.option(
    "--reduce-lr-factor", type=float, default=0.5, help="Factor for ReduceLROnPlateau"
)
@click.option(
    "--min-lr",
    type=float,
    default=1e-7,
    help="Minimum learning rate for ReduceLROnPlateau",
)
@click.option(
    "--scheduler-tmax", type=int, default=50, help="Cosine LR scheduler period (epochs)"
)
@click.option("--no-scheduler", is_flag=True, help="Disable cosine LR scheduler")
@click.option("--no-early-stop", is_flag=True, help="Disable early stopping")
@click.option(
    "--model-path",
    default="track_prediction_ensemble",
    help="Model save path prefix",
)
@click.pass_context
def ensemble(
    ctx,
    num_models,
    epochs,
    batch_size,
    learning_rate,
    early_stop_patience,
    early_stop_min_delta,
    reduce_lr_patience,
    reduce_lr_factor,
    min_lr,
    scheduler_tmax,
    no_scheduler,
    no_early_stop,
    model_path,
):
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
            early_stop_patience=early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
            reduce_lr_patience=reduce_lr_patience,
            reduce_lr_factor=reduce_lr_factor,
            min_lr=min_lr,
            use_scheduler=not no_scheduler,
            scheduler_tmax=scheduler_tmax,
            use_early_stopping=not no_early_stop,
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
@click.option("--max-epochs", default=50, help="Max epochs per trial")
@click.option(
    "--final-epochs", default=50, help="Epochs for final training of best config"
)
@click.option("--batch-size", default=32, help="Batch size")
@click.option(
    "--project-name", default="track_prediction_tuning", help="Tuning project name"
)
@click.option(
    "--model-path",
    default="track_prediction_model_tuned",
    help="Model save path prefix",
)
@click.option("--num-samples", default=20, help="Number of Ray trials")
@click.option("--asha-grace-period", default=5, help="ASHA grace period (epochs)")
@click.option("--asha-reduction-factor", default=3, help="ASHA reduction factor")
@click.option(
    "--ray-dir", default=None, help="Ray local dir for results (default: ~/ray_results)"
)
@click.option(
    "--gpus-per-trial", default=0.0, type=float, help="GPUs per Ray trial (e.g., 1)"
)
@click.option(
    "--cpus-per-trial", default=None, type=float, help="CPUs per Ray trial (default 1)"
)
@click.pass_context
def tune(
    ctx,
    max_epochs,
    final_epochs,
    batch_size,
    project_name,
    model_path,
    num_samples,
    asha_grace_period,
    asha_reduction_factor,
    ray_dir,
    gpus_per_trial,
    cpus_per_trial,
):
    """Tune hyperparameters with Ray Tune (ASHA) and train best model."""
    try:
        full_model_path = create_timestamped_output_dir("tune", model_path)
        tuning_start_time = time.time()

        best_config = tune_hyperparameters_ray(
            data_dir=ctx.obj["data_dir"],
            project_name=project_name,
            max_epochs=max_epochs,
            batch_size=batch_size,
            num_samples=num_samples,
            asha_grace_period=asha_grace_period,
            asha_reduction_factor=asha_reduction_factor,
            directory=ray_dir,
            gpus_per_trial=gpus_per_trial,
            cpus_per_trial=cpus_per_trial,
        )
        tuning_time = time.time() - tuning_start_time

        train_best_model_from_config(
            best_config=best_config,
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
    "tune_hyperparameters_ray",
    "train_best_model_from_config",
    "evaluate_with_cross_validation",
    # Utility functions
    "create_timestamped_output_dir",
    "generate_training_report",
    # CLI
    "cli",
]


if __name__ == "__main__":
    cli()
