"""
Integration tests for InfluxDBResultsClient.
Uses a separate test bucket with setup and cleanup.
Requires InfluxDB 2.x with token authentication.
"""

import pytest
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any

from app.database.influx import (
    InfluxDBResultsClient,
    PredictionResult,
    close_influxdb_client,
)


TEST_URL = os.environ.get("INFLUXDB_URL", "http://127.0.0.1:8086")
TEST_TOKEN = os.environ.get("INFLUXDB_TOKEN", "test-token")
TEST_BUCKET = os.environ.get("INFLUXDB_BUCKET", "ml-buckets")
TEST_ORG = os.environ.get("INFLUXDB_ORG", "tech")


def can_connect():
    """Check if we can connect to InfluxDB 2.x."""
    if not TEST_TOKEN or TEST_TOKEN == "test-token":
        return False
    try:
        client = InfluxDBResultsClient(
            url=TEST_URL,
            token=TEST_TOKEN,
            bucket=TEST_BUCKET,
            org=TEST_ORG,
        )
        result = client.ping()
        client.close()
        return result
    except Exception:
        return False


@pytest.fixture(scope="module")
def influxdb_client():
    """
    Fixture that provides InfluxDB client with test bucket.
    Cleans up after tests.
    """
    if not can_connect():
        pytest.skip(
            "Cannot connect to InfluxDB 2.x - server may not be running or INFLUXDB_TOKEN not set"
        )

    client = None
    try:
        client = InfluxDBResultsClient(
            url=TEST_URL,
            token=TEST_TOKEN,
            bucket=TEST_BUCKET,
            org=TEST_ORG,
        )

        yield client

    finally:
        if client:
            client.close()

    close_influxdb_client()


@pytest.fixture
def sample_predictions() -> List[PredictionResult]:
    """Create sample predictions for testing."""
    base_time = datetime(2026, 2, 15, 10, 0, 0)
    return [
        PredictionResult(
            dataset_name="test_dataset",
            model_name="model_v1",
            timestamp=base_time,
            prediction=26.0,
            actual=25.5,
        ),
        PredictionResult(
            dataset_name="test_dataset",
            model_name="model_v2",
            timestamp=base_time + timedelta(minutes=1),
            prediction=29.5,
            actual=30.0,
        ),
        PredictionResult(
            dataset_name="test_dataset",
            model_name="model_v3",
            timestamp=base_time + timedelta(minutes=2),
            prediction=0.85,
            actual=None,
        ),
    ]


class TestInfluxDBIntegration:
    """Integration tests with real InfluxDB 2.x."""

    def test_ping(self, influxdb_client):
        """Test connection to InfluxDB."""
        result = influxdb_client.ping()
        assert result is True

    def test_write_single_prediction(self, influxdb_client, sample_predictions):
        """Test writing single prediction."""
        result = influxdb_client.write_prediction(sample_predictions[0])
        assert result is True

    def test_write_batch_predictions(self, influxdb_client, sample_predictions):
        """Test writing batch predictions."""
        success_count, failed_count = influxdb_client.write_predictions_batch(
            sample_predictions
        )
        assert success_count == 3
        assert failed_count == 0

    def test_query_all_results(self, influxdb_client, sample_predictions):
        """Test querying all results."""
        influxdb_client.write_predictions_batch(sample_predictions)

        results = influxdb_client.query_results()

        assert len(results) >= 3

    def test_query_with_dataset_filter(self, influxdb_client, sample_predictions):
        """Test querying with dataset filter."""
        influxdb_client.write_predictions_batch(sample_predictions)

        results = influxdb_client.query_results(dataset_name="test_dataset")

        assert len(results) >= 3
        for r in results:
            assert r["dataset_name"] == "test_dataset"

    def test_query_with_model_filter(self, influxdb_client, sample_predictions):
        """Test querying with model filter."""
        influxdb_client.write_predictions_batch(sample_predictions)

        results = influxdb_client.query_results(model_name="model_v1")

        assert len(results) >= 1
        assert results[0]["model_name"] == "model_v1"

    def test_query_with_time_range(self, influxdb_client, sample_predictions):
        """Test querying with time range."""
        influxdb_client.write_predictions_batch(sample_predictions)

        start_time = datetime(2026, 2, 15, 9, 0, 0)
        end_time = datetime(2026, 2, 15, 11, 0, 0)

        results = influxdb_client.query_results(
            start_time=start_time,
            end_time=end_time,
        )

        assert len(results) >= 3

    def test_query_with_limit(self, influxdb_client, sample_predictions):
        """Test querying with limit."""
        influxdb_client.write_predictions_batch(sample_predictions)

        results = influxdb_client.query_results(limit=2)

        assert len(results) == 2

    def test_query_predictions_structure(self, influxdb_client, sample_predictions):
        """Test that query returns correct prediction structure."""
        influxdb_client.write_predictions_batch(sample_predictions)

        results = influxdb_client.query_results(
            dataset_name="test_dataset", model_name="model_v1"
        )

        assert len(results) >= 1
        result = results[0]

        assert "time" in result
        assert "dataset_name" in result
        assert "model_name" in result
        assert "actual" in result
        assert "prediction" in result

        assert result["prediction"] == 26.0

    def test_write_with_none_actual(self, influxdb_client):
        """Test writing prediction without actual value."""
        result = PredictionResult(
            dataset_name="test_no_actual",
            model_name="model_test",
            timestamp=datetime.now(),
            prediction=42.0,
            actual=None,
        )

        success = influxdb_client.write_prediction(result)
        assert success is True

        results = influxdb_client.query_results(dataset_name="test_no_actual")
        assert len(results) >= 1
        assert results[0]["actual"] is None

    def test_write_with_scalar_prediction(self, influxdb_client):
        """Test writing prediction with scalar prediction value."""
        result = PredictionResult(
            dataset_name="test_scalar_pred",
            model_name="model_test",
            timestamp=datetime.now(),
            prediction=0.92,
            actual=100.0,
        )

        success = influxdb_client.write_prediction(result)
        assert success is True

    def test_concurrent_writes(self, influxdb_client):
        """Test multiple sequential writes."""
        results = []
        for i in range(10):
            result = PredictionResult(
                dataset_name="test_concurrent",
                model_name=f"model_{i}",
                timestamp=datetime.now() + timedelta(seconds=i),
                prediction=float(i + 1),
                actual=float(i),
            )
            results.append(result)

        success_count, failed_count = influxdb_client.write_predictions_batch(results)
        assert success_count == 10
        assert failed_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
