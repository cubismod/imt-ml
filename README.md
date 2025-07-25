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

### Hyperparameter Tuning
```bash
# Tune hyperparameters with Hyperband algorithm, then train best model
uv run src/imt_ml/tfrecord_reader.py track_data_export/2024-01-15_12-30-45 tune \
  --max-epochs 50 \
  --final-epochs 100 \
  --project-name my_tuning_project

# Advanced tuning options
uv run src/imt_ml/tfrecord_reader.py track_data_export/2024-01-15_12-30-45 tune \
  --factor 3 \
  --hyperband-iterations 2 \
  --executions-per-trial 3 \
  --distributed
```

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
- **tune**: Uses Keras Tuner's Hyperband algorithm to find optimal hyperparameters, then trains the best model
- **ensemble**: Trains multiple models with different architectures for improved accuracy through ensemble voting
- **cv**: Evaluates model performance using k-fold cross-validation for robust accuracy estimation

### Model Outputs

All training outputs are automatically saved in timestamped subdirectories under `output/` (or custom directory set via `IMT_ML_OUTPUT_DIR`):

- `output/train_20250125_143052/` - Training run outputs
- `output/ensemble_20250125_143052/` - Ensemble training outputs  
- `output/tune_20250125_143052/` - Hyperparameter tuning outputs

Each training mode saves:
- `{model_path}_final.keras` - Final trained model
- `{model_path}_best.keras` - Best model during training (from ModelCheckpoint callback)
- `{model_path}_vocab.json` - Vocabulary mappings for categorical features
- `{model_path}_config.json` - Model configuration and hyperparameters (tune mode only)
- `training_report.md` - **Comprehensive performance and configuration report**

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
