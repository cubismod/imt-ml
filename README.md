# imt-ml

Machine learning component for the "Inky MBTA Tracker" that predicts which tracks/platforms trains will use at MBTA stations. The project exports historical track assignment data from Redis to TensorFlow TFRecord format and trains neural network models for track prediction.

## Quick Start

1. Install dependencies: `uv sync`
2. Export training data: `task export-track-data`
3. Train a model: `uv run src/imt_ml/tfrecord_reader.py path/to/exported/data`

## Data Export

The project exports historical track assignment data from Redis to TensorFlow TFRecord format for model training.

### Export Commands

```bash
# Basic export using environment variables for Redis connection
uv run src/imt_ml/export_track_data.py --output ./track_data_export

# With explicit Redis connection details
uv run src/imt_ml/export_track_data.py \
  --output ./track_data_export \
  --redis-host localhost \
  --redis-port 6379 \
  --redis-password mypassword \
  --records-per-file 5000
```

### Environment Variables

Set these environment variables for Redis connection:
- `IMT_REDIS_ENDPOINT` - Redis host
- `IMT_REDIS_PORT` - Redis port (default: 6379)  
- `IMT_REDIS_PASSWORD` - Redis password

### Data Schema

Each TFRecord contains the following features:

| Feature | Type | Description |
|---------|------|-------------|
| `station_id` | bytes | MBTA station ID |
| `route_id` | bytes | MBTA route ID (e.g., "CR-Franklin") |
| `trip_id` | bytes | MBTA trip ID |
| `headsign` | bytes | Train destination |
| `direction_id` | int64 | Direction (0 or 1) |
| `assignment_type` | bytes | "historical" or "predicted" |
| `track_number` | bytes | **Target variable** - track/platform number |
| `scheduled_timestamp` | float | Unix timestamp of scheduled time |
| `actual_timestamp` | float | Unix timestamp of actual time (0.0 if None) |
| `recorded_timestamp` | float | Unix timestamp when data was recorded |
| `day_of_week` | int64 | Day of week (0=Monday, 6=Sunday) |
| `hour` | int64 | Hour (0-23) |
| `minute` | int64 | Minute (0-59) |

### Export Features

- Only exports data with non-empty track numbers for training
- Handles missing/invalid data gracefully with warnings
- Splits data into multiple TFRecord files for efficient loading
- Includes metadata.json with schema information
- Uses vocabulary tables to handle unknown categories

## TFRecord Reader Usage

The `src/imt_ml/tfrecord_reader.py` module provides comprehensive training capabilities with multiple modes:

### Basic Training
```bash
# Train with default settings
uv run src/imt_ml/tfrecord_reader.py track_data_export/2024-01-15_12-30-45 train

# Custom parameters
uv run src/imt_ml/tfrecord_reader.py track_data_export/2024-01-15_12-30-45 train \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --model-path my_model
```

### Hyperparameter Tuning (Ray Tune ASHA)
```bash
# Tune hyperparameters with Ray Tune's ASHA scheduler, then train best model
uv run src/imt_ml/tfrecord_reader.py track_data_export/2024-01-15_12-30-45 tune \
  --max-epochs 50 \
  --final-epochs 100 \
  --num-samples 40 \
  --project-name my_tuning_project

# Control ASHA behavior and Ray output directory
uv run src/imt_ml/tfrecord_reader.py track_data_export/2024-01-15_12-30-45 tune \
  --max-epochs 75 \
  --num-samples 60 \
  --asha-grace-period 5 \
  --asha-reduction-factor 3 \
  --ray-dir /mnt/shared/ray_results
```



## Distributed Training & Tuning

Two supported setups let you use multiple machines (e.g., your MacBook and a desktop).

### Synchronous Multi-Worker (single trial across machines)

- Runs a single training job distributed across both hosts using TensorFlow `MultiWorkerMirroredStrategy`.
- Requirements:
  - Same Python/TensorFlow/Keras versions on both machines; `export KERAS_BACKEND=tensorflow`.
  - Both machines can read the same dataset path (`--data-dir`) and can reach each other over the network.

Run on each machine, replacing IPs/ports and paths:

```bash
# MacBook (index 0)
uv run src/imt_ml/distributed_launcher.py sync-tune \
  --workers 192.168.1.10:2222,192.168.1.11:2222 \
  --index 0 \
  --data-dir /shared/data \
  --project-name track_tuning

# Desktop (index 1)
uv run src/imt_ml/distributed_launcher.py sync-tune \
  --workers 192.168.1.10:2222,192.168.1.11:2222 \
  --index 1 \
  --data-dir /shared/data \
  --project-name track_tuning
```

Taskfile target (set env vars before running):

```bash
export WORKERS=192.168.1.10:2222,192.168.1.11:2222
export INDEX=0  # or 1 on the desktop
export DATA_DIR=/shared/data
task distributed-sync-tune
```

Notes:
- Worker 0 acts as chief and will train/save the final best model and report.
- Mixed hardware is supported; job speed is limited by the slowest worker.

### Parallel Tuner Trials (multiple trials in parallel)

- Each machine runs separate trials; they share a Keras Tuner state directory to coordinate.
- Faster exploration of hyperparameters; each trial trains locally on that machine.

Run on each machine with a shared directory (NFS/SMB):

```bash
# MacBook (chief)
uv run src/imt_ml/distributed_launcher.py tuner-parallel \
  --tuner-directory /mnt/shared/kt_runs \
  --tuner-id chief \
  --data-dir /shared/data \
  --project-name track_tuning

# Desktop (worker)
uv run src/imt_ml/distributed_launcher.py tuner-parallel \
  --tuner-directory /mnt/shared/kt_runs \
  --tuner-id tuner0 \
  --data-dir /shared/data \
  --project-name track_tuning
```

Taskfile target (set env vars before running):

```bash
export TUNER_DIRECTORY=/mnt/shared/kt_runs
export TUNER_ID=chief  # or tuner0, tuner1, ... on other machines
export DATA_DIR=/shared/data
task distributed-tuner-parallel
```

Notes:
- By default only the `chief` trains/saves the final best model; add `--train-best` on any instance to change that.
- You can combine this with GPUs on the desktop and CPU-only on the Mac; each trial uses whatever hardware is available locally.

### Ensemble Training
```bash
# Train multiple models for improved accuracy
uv run src/imt_ml/tfrecord_reader.py track_data_export/2024-01-15_12-30-45 ensemble \
  --num-models 5 \
  --epochs 50

# Custom ensemble settings
uv run src/imt_ml/tfrecord_reader.py track_data_export/2024-01-15_12-30-45 ensemble \
  --num-models 3 \
  --epochs 75 \
  --learning-rate 0.0005 \
  --model-path ensemble_model

# Train longer and relax early stopping
uv run src/imt_ml/tfrecord_reader.py track_data_export/<date> ensemble \
  --epochs 150 \
  --early-stop-patience 20 \
  --early-stop-min-delta 1e-5 \
  --reduce-lr-patience 10 \
  --reduce-lr-factor 0.5 \
  --scheduler-tmax 100

# Disable early stopping or the cosine scheduler
uv run src/imt_ml/tfrecord_reader.py track_data_export/<date> ensemble --no-early-stop
uv run src/imt_ml/tfrecord_reader.py track_data_export/<date> ensemble --no-scheduler
```

### Cross-Validation
```bash
# Evaluate model performance with k-fold cross-validation
uv run src/imt_ml/tfrecord_reader.py track_data_export/2024-01-15_12-30-45 cv \
  --k-folds 10 \
  --epochs 30
```

### Training Modes Explained

- **train**: Standard training with fixed hyperparameters, includes regularization and data augmentation for small datasets
- **tune**: Uses Ray Tune's ASHA scheduler to find optimal hyperparameters, then trains the best model
- **ensemble**: Trains multiple models with different architectures for improved accuracy through ensemble voting
- **cv**: Evaluates model performance using k-fold cross-validation for robust accuracy estimation

### Model Outputs

All training outputs are automatically saved in timestamped subdirectories under `output/` (or custom directory set via `IMT_ML_OUTPUT_DIR`):

- `output/train_20250125_143052/` - Training run outputs
- `output/ensemble_20250125_143052/` - Ensemble training outputs  
- `output/tune_20250125_143052/` - Hyperparameter tuning outputs

Each training mode saves:
- `{model_path}_final.keras` - Final trained model
- `{model_path}_best.keras` - Best model during training
- `{model_path}_vocab.json` - Vocabulary mappings for categorical features
- `{model_path}_config.json` - Model configuration and hyperparameters (tune mode only)
- `training_report.md` - **Comprehensive performance and configuration report**

Loading notes:
- For inference, load checkpoints with `compile=False` to skip optimizer restore.
- During tuning, best checkpoints include optimizer state to support resume; for inference use `compile=False`.

#### Output Directory Configuration

Set the `IMT_ML_OUTPUT_DIR` environment variable to customize the base output directory:

```bash
# Use custom output directory
export IMT_ML_OUTPUT_DIR="/path/to/my/models"
uv run src/imt_ml/tfrecord_reader.py data/ train

# Outputs will be saved to: /path/to/my/models/train_20250125_143052/
```

#### Training Reports

Each training session automatically generates a detailed `training_report.md` file containing:

- **Performance Metrics**: Final validation accuracy, loss, and training history
- **Training Configuration**: All hyperparameters, model architecture details
- **Dataset Information**: Vocabulary sizes, record counts, data splits
- **Timing Information**: Training duration and timestamps
- **Model Details**: Parameter count, regularization settings
- **Command-Specific Info**: 
  - **train**: Regularization and data augmentation details
  - **ensemble**: Individual model performance and ensemble statistics
  - **tune**: Hyperparameter search results and optimization details
  - **cv**: Cross-validation fold results and confidence intervals

Example report structure:
```markdown
# Training Report - Train
Generated: 2025-01-25 14:30:52

## Overview
- Command: train
- Training Duration: 245.3 seconds (4.1 minutes)
- Output Directory: output/train_20250125_143052

## Final Performance Metrics
- Validation Loss: 0.3247
- Validation Accuracy: 0.8756
- Best Validation Accuracy: 0.8834

## Training Configuration
- Epochs: 50
- Batch Size: 32
- Learning Rate: 0.001
- Dataset Size: 15,247 records
```

### Features

- **Automatic vocabulary building** from TFRecord data
- **Cyclical time encoding** (sin/cos) for temporal features
- **Class weight balancing** for imbalanced datasets
- **Data augmentation** for small datasets (<10k samples)
- **Progress bars** for all operations
- **Early stopping** and learning rate scheduling
- **Regularization** (L1/L2, dropout, batch normalization)

## Model Architecture & Feature Engineering

The machine learning pipeline applies the following transformations:

1. **Categorical Encoding**: Station, route, and track IDs converted to integer indices
2. **Cyclical Time Features**: Hour, minute, and day-of-week converted to sin/cos pairs
3. **Embeddings**: Categorical features use learned embeddings
4. **Normalization**: Timestamps are normalized

### Model Components

- Embedding layers for categorical features (stations, routes, directions)
- Dense layers with dropout for regularization  
- Softmax output for track classification
- Cyclical time encoding to capture temporal patterns
- Batch normalization for training stability

### Programmatic Usage

```python
from src.imt_ml.tfrecord_reader import load_tfrecord_dataset, create_simple_model

# Load the dataset
train_ds, val_ds, vocab_info = load_tfrecord_dataset('./track_data_export')

# Create and train a model
model = create_simple_model(vocab_info)
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds, validation_data=val_ds, epochs=10)
```

## Project Docs

For how to install uv and Python, see [installation.md](installation.md).

For development workflows, see [development.md](development.md).

For instructions on publishing to PyPI, see [publishing.md](publishing.md).

* * *

*This project was built from
[simple-modern-uv](https://github.com/jlevy/simple-modern-uv).*
