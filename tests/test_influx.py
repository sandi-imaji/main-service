"""
Unit tests for InfluxDBResultsClient.
Uses mocking to avoid connecting to actual InfluxDB.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List

from app.database.influx import (
    InfluxDBResultsClient,
    PredictionResult,
    create_influxdb_client_from_config,
    get_influxdb_client,
    close_influxdb_client,
)


@pytest.fixture
def mock_influxdb_client():
    """Create a mock InfluxDB 2.x client."""
    with patch("app.database.influx.InfluxDBClient") as mock_client:
        mock_instance = MagicMock()
        mock_write_api = MagicMock()
        mock_query_api = MagicMock()

        mock_instance.write_api.return_value = mock_write_api
        mock_instance.query_api.return_value = mock_query_api
        mock_instance.ping.return_value = True

        mock_client.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def client(mock_influxdb_client):
    """Create InfluxDBResultsClient with mocked backend."""
    with patch("app.database.influx.InfluxDBClient") as mock_client:
        mock_client.return_value = mock_influxdb_client
        client = InfluxDBResultsClient(
            url="http://localhost:8086",
            token="test_token",
            bucket="ml_predictions",
            org="smart-ai",
        )
        yield client


@pytest.fixture
def sample_prediction_result():
    """Create sample PredictionResult for testing."""
    return PredictionResult(
        dataset_name="test_dataset",
        model_name="model_v1",
        timestamp=datetime(2026, 2, 15, 10, 30, 0),
        prediction=26.0,
        actual=25.5,
    )


@pytest.fixture
def sample_prediction_results():
    """Create list of sample PredictionResults."""
    return [
        PredictionResult(
            dataset_name="test_dataset",
            model_name="model_v1",
            timestamp=datetime(2026, 2, 15, 10, 30, 0),
            prediction=26.0,
            actual=25.5,
        ),
        PredictionResult(
            dataset_name="test_dataset",
            model_name="model_v2",
            timestamp=datetime(2026, 2, 15, 10, 31, 0),
            prediction=29.5,
            actual=30.0,
        ),
        PredictionResult(
            dataset_name="another_dataset",
            model_name="model_v3",
            timestamp=datetime(2026, 2, 15, 10, 32, 0),
            prediction=0.85,
            actual=None,
        ),
    ]


class TestPredictionResult:
    """Tests for PredictionResult dataclass."""

    def test_prediction_result_creation(self):
        """Test creating PredictionResult with all fields."""
        result = PredictionResult(
            dataset_name="test_dataset",
            model_name="model_v1",
            timestamp=datetime.now(),
            prediction=26.0,
            actual=25.5,
        )

        assert result.dataset_name == "test_dataset"
        assert result.model_name == "model_v1"
        assert result.prediction == 26.0
        assert result.actual == 25.5

    def test_prediction_result_with_scalar_prediction(self):
        """Test PredictionResult with scalar prediction value."""
        result = PredictionResult(
            dataset_name="test",
            model_name="model_v2",
            timestamp=datetime.now(),
            prediction=0.95,
            actual=10.0,
        )

        assert isinstance(result.prediction, float)
        assert result.prediction == 0.95

    def test_prediction_result_with_none_actual(self):
        """Test PredictionResult with None actual value."""
        result = PredictionResult(
            dataset_name="test",
            model_name="model_v1",
            timestamp=datetime.now(),
            prediction=26.0,
            actual=None,
        )

        assert result.actual is None


class TestInfluxDBResultsClient:
    """Tests for InfluxDBResultsClient class."""

    def test_client_initialization(self, mock_influxdb_client):
        """Test client initialization with default parameters."""
        client = InfluxDBResultsClient(
            url="http://localhost:8086",
            token="test_token",
            bucket="ml_predictions",
            org="smart-ai",
        )

        assert client.url == "http://localhost:8086"
        assert client.bucket == "ml_predictions"
        assert client.retries == 3
        assert client.token == "test_token"
        assert client.org == "smart-ai"

    def test_client_initialization_custom_params(self, mock_influxdb_client):
        """Test client initialization with custom parameters."""
        client = InfluxDBResultsClient(
            url="http://192.168.1.100:8086",
            token="custom_token",
            bucket="custom_bucket",
            org="custom_org",
            timeout=60,
            retries=5,
        )

        assert client.url == "http://192.168.1.100:8086"
        assert client.bucket == "custom_bucket"
        assert client.timeout == 60
        assert client.retries == 5

    def test_create_point(self, client, sample_prediction_result):
        """Test point creation from PredictionResult."""
        point = client._create_point(sample_prediction_result)

        assert point._name == "results"
        assert point._tags["dataset_name"] == "test_dataset"
        assert point._tags["model_name"] == "model_v1"

    def test_create_point_with_scalar_prediction(self, client):
        """Test point creation with scalar prediction value."""
        result = PredictionResult(
            dataset_name="test",
            model_name="model_v1",
            timestamp=datetime.now(),
            prediction=0.85,
            actual=25.5,
        )

        point = client._create_point(result)

        assert "prediction" in point._fields

    def test_create_point_with_int_prediction(self, client):
        """Test point creation with integer prediction value."""
        result = PredictionResult(
            dataset_name="test",
            model_name="model_v1",
            timestamp=datetime.now(),
            prediction=26,
            actual=25,
        )

        point = client._create_point(result)

        assert "prediction" in point._fields

    def test_create_point_with_none_actual(self, client):
        """Test point creation without actual value."""
        result = PredictionResult(
            dataset_name="test",
            model_name="model_v1",
            timestamp=datetime.now(),
            prediction=26.0,
            actual=None,
        )

        point = client._create_point(result)
        assert "prediction" in point._fields

    def test_create_point_with_bool_prediction(self, client):
        """Test point creation with boolean prediction."""
        result = PredictionResult(
            dataset_name="test",
            model_name="model_v1",
            timestamp=datetime.now(),
            prediction=True,
            actual=True,
        )

        point = client._create_point(result)

        assert "prediction" in point._fields

    def test_write_prediction_success(
        self, client, sample_prediction_result, mock_influxdb_client
    ):
        """Test successful single prediction write."""
        mock_write_api = mock_influxdb_client.write_api.return_value
        mock_write_api.write.return_value = True

        result = client.write_prediction(sample_prediction_result)

        assert result is True
        mock_write_api.write.assert_called_once()

    def test_write_prediction_failure(
        self, client, sample_prediction_result, mock_influxdb_client
    ):
        """Test failed prediction write."""
        mock_write_api = mock_influxdb_client.write_api.return_value
        mock_write_api.write.side_effect = Exception("Connection error")

        result = client.write_prediction(sample_prediction_result)

        assert result is False

    def test_write_predictions_batch_success(
        self, client, sample_prediction_results, mock_influxdb_client
    ):
        """Test successful batch write."""
        mock_write_api = mock_influxdb_client.write_api.return_value
        mock_write_api.write.return_value = True

        success_count, failed_count = client.write_predictions_batch(
            sample_prediction_results
        )

        assert success_count == 3
        assert failed_count == 0

    def test_write_predictions_batch_empty_list(self, client):
        """Test batch write with empty list."""
        success_count, failed_count = client.write_predictions_batch([])

        assert success_count == 0
        assert failed_count == 0

    def test_write_predictions_batch_with_retry(
        self, client, sample_prediction_results, mock_influxdb_client
    ):
        """Test batch write with retry on failure."""
        call_count = 0

        def write_with_fail(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary error")
            return True

        mock_write_api = mock_influxdb_client.write_api.return_value
        mock_write_api.write.side_effect = write_with_fail

        success_count, failed_count = client.write_predictions_batch(
            sample_prediction_results
        )

        assert success_count == 3
        assert call_count == 2

    def test_write_predictions_batch_max_retries_exceeded(
        self, client, sample_prediction_results, mock_influxdb_client
    ):
        """Test batch write with retries exceeded."""
        mock_write_api = mock_influxdb_client.write_api.return_value
        mock_write_api.write.side_effect = Exception("Persistent error")

        success_count, failed_count = client.write_predictions_batch(
            sample_prediction_results
        )

        assert success_count == 0
        assert failed_count == 3

    def test_query_results_success(self, client, mock_influxdb_client):
        """Test successful query."""
        # Create mock Flux record
        mock_record = MagicMock()
        mock_record.get_time.return_value = datetime(2026, 2, 15, 10, 30, 0)
        mock_record.values = {
            "dataset_name": "test_dataset",
            "model_name": "model_v1",
            "actual": 25.5,
            "prediction": 26.0,
        }

        mock_table = MagicMock()
        mock_table.records = [mock_record]

        mock_query_api = mock_influxdb_client.query_api.return_value
        mock_query_api.query.return_value = [mock_table]

        results = client.query_results()

        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]["dataset_name"] == "test_dataset"

    def test_query_results_with_filters(self, client, mock_influxdb_client):
        """Test query with filters."""
        mock_table = MagicMock()
        mock_table.records = []

        mock_query_api = mock_influxdb_client.query_api.return_value
        mock_query_api.query.return_value = [mock_table]

        results = client.query_results(
            dataset_name="test_dataset",
            model_name="model_v1",
            start_time=datetime(2026, 2, 1),
            end_time=datetime(2026, 2, 15),
            limit=100,
        )

        mock_query_api.query.assert_called_once()
        call_args = mock_query_api.query.call_args
        assert "test_dataset" in str(call_args)

    def test_query_results_empty_result(self, client, mock_influxdb_client):
        """Test query with empty result."""
        mock_query_api = mock_influxdb_client.query_api.return_value
        mock_query_api.query.return_value = []

        results = client.query_results()

        assert results == []

    def test_query_results_exception(self, client, mock_influxdb_client):
        """Test query with exception."""
        mock_query_api = mock_influxdb_client.query_api.return_value
        mock_query_api.query.side_effect = Exception("Query error")

        results = client.query_results()

        assert results == []

    def test_ping_success(self, client, mock_influxdb_client):
        """Test successful ping."""
        mock_influxdb_client.ping.return_value = True

        result = client.ping()

        assert result is True
        mock_influxdb_client.ping.assert_called_once()

    def test_ping_failure(self, client, mock_influxdb_client):
        """Test failed ping."""
        mock_influxdb_client.ping.side_effect = Exception("Connection error")

        result = client.ping()

        assert result is False

    def test_close(self, client, mock_influxdb_client):
        """Test close method."""
        mock_write_api = mock_influxdb_client.write_api.return_value

        client.close()

        mock_write_api.close.assert_called_once()
        mock_influxdb_client.close.assert_called_once()

    def test_flush(self, client, mock_influxdb_client):
        """Test flush method."""
        mock_write_api = mock_influxdb_client.write_api.return_value

        client.flush()

        mock_write_api.flush.assert_called_once()


class TestFactoryFunctions:
    """Tests for factory functions."""

    @patch("app.database.influx.InfluxDBResultsClient")
    def test_create_influxdb_client_from_config(self, mock_client_class):
        """Test creating client from config."""
        mock_config = Mock()
        mock_config.influxdb_url = "http://localhost:8086"
        mock_config.influxdb_token = "test_token"
        mock_config.influxdb_bucket = "ml_predictions"
        mock_config.influxdb_org = "smart-ai"

        client = create_influxdb_client_from_config(mock_config)

        mock_client_class.assert_called_once()
        call_kwargs = mock_client_class.call_args[1]
        assert call_kwargs["url"] == "http://localhost:8086"
        assert call_kwargs["token"] == "test_token"
        assert call_kwargs["bucket"] == "ml_predictions"
        assert call_kwargs["org"] == "smart-ai"

    @patch("app.database.influx.InfluxDBResultsClient")
    def test_create_influxdb_client_with_https(self, mock_client_class):
        """Test creating client with HTTPS URL."""
        mock_config = Mock()
        mock_config.influxdb_url = "https://influxdb.example.com:8086"

        client = create_influxdb_client_from_config(mock_config)

        call_kwargs = mock_client_class.call_args[1]
        assert call_kwargs["url"] == "https://influxdb.example.com:8086"


class TestSingletonPattern:
    """Tests for singleton pattern."""

    def test_get_influxdb_client_singleton(self):
        """Test singleton returns same instance."""
        close_influxdb_client()

        with patch("app.database.influx.InfluxDBClient"):
            client1 = get_influxdb_client()
            client2 = get_influxdb_client()

            assert client1 is client2

    def test_close_influxdb_client_resets_singleton(self):
        """Test close resets singleton."""
        with patch("app.database.influx.InfluxDBClient"):
            client1 = get_influxdb_client()
            close_influxdb_client()


class TestIntegrationScenarios:
    """Integration-style tests with mocked components."""

    def test_write_multiple_predictions_different_types(
        self, client, mock_influxdb_client
    ):
        """Test writing predictions with different data types."""
        mock_write_api = mock_influxdb_client.write_api.return_value
        mock_write_api.write.return_value = True

        results = [
            PredictionResult(
                dataset_name="test",
                model_name="model_v1",
                timestamp=datetime.now(),
                prediction=26.0,
                actual=25.5,
            ),
            PredictionResult(
                dataset_name="test",
                model_name="model_v2",
                timestamp=datetime.now(),
                prediction=0.85,
                actual=None,
            ),
            PredictionResult(
                dataset_name="test",
                model_name="model_v3",
                timestamp=datetime.now(),
                prediction=False,
                actual=True,
            ),
        ]

        success_count, failed_count = client.write_predictions_batch(results)

        assert success_count == 3
        assert failed_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
