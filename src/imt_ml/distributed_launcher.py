"""
Distributed launcher for synchronous multi-worker training and parallel tuner trials.

Usage examples:

Synchronous distributed tuning (both machines run the same command):
  uv run python -m imt_ml.distributed_launcher sync-tune \
      --workers 192.168.1.10:2222,192.168.1.11:2222 --index 0 \
      --data-dir /shared/data --project-name track_tuning

  uv run python -m imt_ml.distributed_launcher sync-tune \
      --workers 192.168.1.10:2222,192.168.1.11:2222 --index 1 \
      --data-dir /shared/data --project-name track_tuning

Parallel hyperparameter tuning (each machine runs separate trials; shared directory):
  uv run python -m imt_ml.distributed_launcher tuner-parallel \
      --tuner-directory /mnt/shared/kt_runs --tuner-id chief \
      --data-dir /shared/data --project-name track_tuning

  uv run python -m imt_ml.distributed_launcher tuner-parallel \
      --tuner-directory /mnt/shared/kt_runs --tuner-id tuner0 \
      --data-dir /shared/data --project-name track_tuning
"""

from __future__ import annotations

import json
import os
from typing import List

import click

from imt_ml.reporting import create_timestamped_output_dir, generate_training_report
from imt_ml.training import train_best_model, tune_hyperparameters


def _set_tf_config(workers: List[str], index: int) -> None:
    cluster = {"worker": workers}
    tf_config = {"cluster": cluster, "task": {"type": "worker", "index": index}}
    os.environ["TF_CONFIG"] = json.dumps(tf_config)


@click.group()
def cli() -> None:
    """Distributed launch helpers for training/tuning."""


@cli.command("sync-tune")
@click.option(
    "--workers", required=True, help="Comma-separated host:port list for workers"
)
@click.option("--index", type=int, required=True, help="This worker's index (0..N-1)")
@click.option("--data-dir", required=True, type=click.Path(exists=True))
@click.option("--project-name", default="track_prediction_tuning")
@click.option("--max-epochs", default=50, type=int)
@click.option("--factor", default=3, type=int)
@click.option("--hyperband-iterations", default=1, type=int)
@click.option("--executions-per-trial", default=2, type=int)
@click.option("--batch-size", default=32, type=int)
@click.option("--model-path", default="track_prediction_model_tuned")
def sync_tune(
    workers: str,
    index: int,
    data_dir: str,
    project_name: str,
    max_epochs: int,
    factor: int,
    hyperband_iterations: int,
    executions_per_trial: int,
    batch_size: int,
    model_path: str,
) -> None:
    """Run synchronous distributed tuning across the provided workers."""
    worker_list = [w.strip() for w in workers.split(",") if w.strip()]
    if index < 0 or index >= len(worker_list):
        raise click.ClickException("--index must be within range of --workers")

    _set_tf_config(worker_list, index)

    # Run tuning with distributed strategy enabled
    full_model_path = create_timestamped_output_dir("tune", model_path)
    tuner = tune_hyperparameters(
        data_dir=data_dir,
        project_name=project_name,
        max_epochs=max_epochs,
        factor=factor,
        hyperband_iterations=hyperband_iterations,
        batch_size=batch_size,
        distributed=True,
        executions_per_trial=executions_per_trial,
    )

    # Only worker 0 acts as chief to train best model and write reports
    if index == 0:
        train_best_model(
            tuner=tuner,
            data_dir=data_dir,
            epochs=max_epochs,
            batch_size=batch_size,
            model_save_path=full_model_path,
            tuning_time=0.0,
            generate_report_func=generate_training_report,
        )


@cli.command("tuner-parallel")
@click.option("--tuner-directory", required=True, type=click.Path())
@click.option(
    "--tuner-id", required=True, help="Unique ID (e.g., chief, tuner0, tuner1)"
)
@click.option("--data-dir", required=True, type=click.Path(exists=True))
@click.option("--project-name", default="track_prediction_tuning")
@click.option("--max-epochs", default=50, type=int)
@click.option("--factor", default=3, type=int)
@click.option("--hyperband-iterations", default=1, type=int)
@click.option("--executions-per-trial", default=2, type=int)
@click.option("--batch-size", default=32, type=int)
@click.option("--model-path", default="track_prediction_model_tuned")
@click.option(
    "--train-best/--no-train-best",
    default=None,
    help="Whether to train best model here; defaults to only on chief",
)
def tuner_parallel(
    tuner_directory: str,
    tuner_id: str,
    data_dir: str,
    project_name: str,
    max_epochs: int,
    factor: int,
    hyperband_iterations: int,
    executions_per_trial: int,
    batch_size: int,
    model_path: str,
    train_best: bool | None,
) -> None:
    """Run parallel tuner trials across machines sharing a directory."""
    os.environ["KERASTUNER_TUNER_ID"] = tuner_id

    full_model_path = create_timestamped_output_dir("tune", model_path)
    tuner = tune_hyperparameters(
        data_dir=data_dir,
        project_name=project_name,
        max_epochs=max_epochs,
        factor=factor,
        hyperband_iterations=hyperband_iterations,
        batch_size=batch_size,
        distributed=False,
        executions_per_trial=executions_per_trial,
        directory=tuner_directory,
    )

    should_train_best = train_best if train_best is not None else (tuner_id == "chief")
    if should_train_best:
        train_best_model(
            tuner=tuner,
            data_dir=data_dir,
            epochs=max_epochs,
            batch_size=batch_size,
            model_save_path=full_model_path,
            tuning_time=0.0,
            generate_report_func=generate_training_report,
        )


if __name__ == "__main__":
    cli()


# Convenience entrypoints to call subcommands directly via console scripts
def sync_tune_cli() -> None:
    """Console entry for `distributed-sync-tune`.

    Invokes the `sync-tune` subcommand so users can run:
      uv run distributed-sync-tune --workers ... --index ...
    """
    # Use the click command object to parse argv for this subcommand
    cli.commands["sync-tune"]()  # type: ignore[misc]


def tuner_parallel_cli() -> None:
    """Console entry for `distributed-tuner-parallel`.

    Invokes the `tuner-parallel` subcommand so users can run:
      uv run distributed-tuner-parallel --tuner-directory ... --tuner-id ...
    """
    cli.commands["tuner-parallel"]()  # type: ignore[misc]
