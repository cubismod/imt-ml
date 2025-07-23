# MBTA Track Data Export for TensorFlow

This directory contains scripts to export historical track assignment data from Redis into TensorFlow TFRecord format for machine learning model training.

## Files

- `export_track_data.py` - Main export script that connects to Redis and exports data
- `tfrecord_reader.py` - TensorFlow utilities for reading and preprocessing the exported data
- `README_EXPORT.md` - This file

## Usage

### 1. Export Data from Redis

```bash
# Basic usage (uses environment variables for Redis connection)
python export_track_data.py --output ./my_export

# With explicit Redis connection details
python export_track_data.py \
  --output ./my_export \
  --redis-host localhost \
  --redis-port 6379 \
  --redis-password mypassword \
  --records-per-file 5000
```

### Environment Variables
- `IMT_REDIS_ENDPOINT` - Redis host
- `IMT_REDIS_PORT` - Redis port (default: 6379)  
- `IMT_REDIS_PASSWORD` - Redis password

### 2. Use Exported Data in TensorFlow

```python
from tfrecord_reader import load_tfrecord_dataset, create_simple_model

# Load the dataset
train_ds, val_ds, vocab_info = load_tfrecord_dataset('./my_export')

# Create and train a model
model = create_simple_model(vocab_info)
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds, validation_data=val_ds, epochs=10)
```

## Data Schema

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

## Feature Engineering

The `tfrecord_reader.py` applies the following transformations:

1. **Categorical Encoding**: Station, route, and track IDs converted to integer indices
2. **Cyclical Time Features**: Hour, minute, and day-of-week converted to sin/cos pairs
3. **Embeddings**: Categorical features use learned embeddings
4. **Normalization**: Timestamps are normalized

## Model Architecture

The example model includes:
- Embedding layers for categorical features
- Dense layers with dropout for regularization  
- Softmax output for track classification
- Cyclical time encoding to capture temporal patterns

## Requirements

```bash
pip install tensorflow redis pydantic
```

## Notes

- Only exports data with non-empty track numbers for training
- Handles missing/invalid data gracefully with warnings
- Splits data into multiple TFRecord files for efficient loading
- Includes metadata.json with schema information
- Uses vocabulary tables to handle unknown categories