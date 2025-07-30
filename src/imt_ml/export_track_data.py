"""
Export historical track assignment data from Redis to TensorFlow TFRecord format.

This script connects to the Redis instance used by the track predictor and exports
all historical track assignment data into TFRecord files suitable for TensorFlow
training.
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import redis.asyncio as redis
import tensorflow as tf
from pydantic import ValidationError
from tqdm import tqdm

from imt_ml.shared_types.shared_types import TrackAssignment


class TrackDataExporter:
    """Export historical track assignment data to TFRecord format."""

    def __init__(self, redis_host: str, redis_port: int, redis_password: str):
        """Initialize the exporter with Redis connection details."""
        self.redis: redis.Redis = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            decode_responses=False,  # Keep as bytes for key scanning
        )

    async def get_all_timeseries_keys(self) -> list[str]:
        """Get all track timeseries keys from Redis."""
        keys: list[str] = []
        async for key in self.redis.scan_iter(match="track_timeseries:*"):
            key_str = key.decode("utf-8") if isinstance(key, bytes) else str(key)
            keys.append(key_str)
        return keys

    async def get_historical_assignments_for_key(
        self, timeseries_key: str
    ) -> list[TrackAssignment]:
        """Get all historical assignments for a given timeseries key."""
        assignments: list[TrackAssignment] = []

        # Get all assignment keys from the sorted set
        assignment_keys = await self.redis.zrange(timeseries_key, 0, -1)

        for assignment_key in assignment_keys:
            assignment_key_str = (
                assignment_key.decode("utf-8")
                if isinstance(assignment_key, bytes)
                else str(assignment_key)
            )
            assignment_data = await self.redis.get(assignment_key_str)

            if assignment_data:
                try:
                    assignment = TrackAssignment.model_validate_json(assignment_data)
                    # remove noninteger values due to an earlier bug
                    if (
                        assignment.track_number
                        and not assignment.track_number.isdigit()
                    ):
                        print(f"Skipping invalid track num: {assignment.track_number}")
                        continue
                    assignments.append(assignment)
                except ValidationError as e:
                    print(
                        f"Warning: Failed to parse assignment {assignment_key_str}: {e}"
                    )
                    continue

        return assignments

    async def export_all_data(self) -> list[TrackAssignment]:
        """Export all historical track assignment data from Redis."""
        print("Scanning for timeseries keys...")
        timeseries_keys = await self.get_all_timeseries_keys()
        print(f"Found {len(timeseries_keys)} timeseries keys")

        all_assignments: list[TrackAssignment] = []

        for key in tqdm(timeseries_keys, desc="Processing timeseries keys", unit="key"):
            assignments = await self.get_historical_assignments_for_key(key)
            all_assignments.extend(assignments)
            tqdm.write(f"Found {len(assignments)} assignments for {key}")

        print(f"Total assignments collected: {len(all_assignments)}")
        return all_assignments

    def create_tf_example(self, assignment: TrackAssignment) -> tf.train.Example:
        """Convert a TrackAssignment to a TensorFlow Example."""
        # Encode categorical features
        station_id_bytes = assignment.station_id.encode("utf-8")
        route_id_bytes = assignment.route_id.encode("utf-8")
        trip_id_bytes = assignment.trip_id.encode("utf-8")
        headsign_bytes = assignment.headsign.encode("utf-8")
        assignment_type_bytes = assignment.assignment_type.value.encode("utf-8")
        track_number_bytes = (assignment.track_number or "").encode("utf-8")

        # Convert timestamps to unix timestamps (float)
        scheduled_timestamp = assignment.scheduled_time.timestamp()
        actual_timestamp = (
            assignment.actual_time.timestamp() if assignment.actual_time else 0.0
        )
        recorded_timestamp = assignment.recorded_time.timestamp()

        # Create feature dictionary
        feature: dict[str, tf.train.Feature] = {
            "station_id": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[station_id_bytes])
            ),
            "route_id": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[route_id_bytes])
            ),
            "trip_id": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[trip_id_bytes])
            ),
            "headsign": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[headsign_bytes])
            ),
            "direction_id": tf.train.Feature(
                int64_list=tf.train.Int64List(value=[assignment.direction_id])
            ),
            "assignment_type": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[assignment_type_bytes])
            ),
            "track_number": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[track_number_bytes])
            ),
            "scheduled_timestamp": tf.train.Feature(
                float_list=tf.train.FloatList(value=[scheduled_timestamp])
            ),
            "actual_timestamp": tf.train.Feature(
                float_list=tf.train.FloatList(value=[actual_timestamp])
            ),
            "recorded_timestamp": tf.train.Feature(
                float_list=tf.train.FloatList(value=[recorded_timestamp])
            ),
            "day_of_week": tf.train.Feature(
                int64_list=tf.train.Int64List(value=[assignment.day_of_week])
            ),
            "hour": tf.train.Feature(
                int64_list=tf.train.Int64List(value=[assignment.hour])
            ),
            "minute": tf.train.Feature(
                int64_list=tf.train.Int64List(value=[assignment.minute])
            ),
        }

        return tf.train.Example(features=tf.train.Features(feature=feature))

    def write_tfrecords(
        self,
        assignments: list[TrackAssignment],
        output_path: str,
        records_per_file: int = 10000,
    ) -> None:
        """Write assignments to TFRecord files."""
        output_dir: Path = Path(output_path) / datetime.now().strftime(
            "%Y-%m-%d_%H%M%S"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write metadata file
        metadata = {
            "total_records": len(assignments),
            "created_at": datetime.now().isoformat(),
            "schema": {
                "station_id": "bytes (string)",
                "route_id": "bytes (string)",
                "trip_id": "bytes (string)",
                "headsign": "bytes (string)",
                "direction_id": "int64",
                "assignment_type": "bytes (string)",
                "track_number": "bytes (string, target variable)",
                "scheduled_timestamp": "float (unix timestamp)",
                "actual_timestamp": "float (unix timestamp, 0.0 if None)",
                "recorded_timestamp": "float (unix timestamp)",
                "day_of_week": "int64 (0=Monday, 6=Sunday)",
                "hour": "int64 (0-23)",
                "minute": "int64 (0-59)",
            },
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Write TFRecord files
        file_count = 0
        total_files = (len(assignments) + records_per_file - 1) // records_per_file

        for i in tqdm(
            range(0, len(assignments), records_per_file),
            desc="Writing TFRecord files",
            unit="file",
            total=total_files,
        ):
            file_count += 1
            batch = assignments[i : i + records_per_file]
            filename = output_dir / f"track_data_{file_count:04d}.tfrecord"

            tqdm.write(f"Writing {len(batch)} records to {filename}")

            writer = tf.io.TFRecordWriter(str(filename))
            try:
                for assignment in tqdm(
                    batch,
                    desc=f"Creating examples for file {file_count}",
                    unit="record",
                    leave=False,
                ):
                    try:
                        example = self.create_tf_example(assignment)
                        writer.write(example.SerializeToString())
                    except Exception as e:
                        tqdm.write(f"Warning: Failed to serialize assignment: {e}")
                        continue
            finally:
                writer.close()

        print(f"Wrote {len(assignments)} records to {file_count} TFRecord files")
        print(f"Files saved to: {output_dir}")

    async def close(self) -> None:
        """Close the Redis connection."""
        await self.redis.aclose()


async def main() -> None:
    """Main function to export track data."""
    parser = argparse.ArgumentParser(description="Export track data to TFRecord format")
    parser.add_argument(
        "--output",
        "-o",
        default="./track_data_export",
        help="Output directory for TFRecord files",
    )
    parser.add_argument(
        "--redis-host",
        default=os.environ.get("IMT_REDIS_ENDPOINT", "localhost"),
        help="Redis host",
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=int(os.environ.get("IMT_REDIS_PORT", "6379")),
        help="Redis port",
    )
    parser.add_argument(
        "--redis-password",
        default=os.environ.get("IMT_REDIS_PASSWORD", ""),
        help="Redis password",
    )
    parser.add_argument(
        "--records-per-file",
        type=int,
        default=10000,
        help="Number of records per TFRecord file",
    )

    args = parser.parse_args()

    # Validate Redis connection details
    if not args.redis_host:
        print(
            "Error: Redis host is required. Set IMT_REDIS_ENDPOINT environment variable or use --redis-host"
        )
        sys.exit(1)

    exporter = TrackDataExporter(args.redis_host, args.redis_port, args.redis_password)

    try:
        print("Connecting to Redis...")
        await exporter.redis.ping()
        print("Connected successfully!")

        print("Exporting data...")
        assignments = await exporter.export_all_data()

        if not assignments:
            print("No track assignments found in Redis")
            return

        print(f"Writing {len(assignments)} assignments to TFRecord files...")
        exporter.write_tfrecords(assignments, args.output, args.records_per_file)

        print("Export completed successfully!")

    except Exception as e:
        print(f"Error during export: {e}")
        sys.exit(1)
    finally:
        await exporter.close()


def run() -> None:
    asyncio.run(main())
