"""
Shared fixtures for core module tests with database integration.
Setup/teardown per module using SQLite in-memory.
"""

import pytest
import random
from sqlmodel import SQLModel, create_engine, Session
from sqlalchemy.pool import StaticPool
from unittest.mock import Mock, MagicMock

# In-memory SQLite database for testing
TEST_DATABASE_URL = "sqlite:///:memory:"


@pytest.fixture(scope="module")
def db_engine():
  """
  Create a fresh database engine for each test module.
  This provides isolation between test modules.
  """
  engine = create_engine(
      TEST_DATABASE_URL,
      connect_args={"check_same_thread": False},
      poolclass=StaticPool,
  )
  SQLModel.metadata.create_all(engine)
  yield engine
  SQLModel.metadata.drop_all(engine)


@pytest.fixture(scope="function")
def db_session(db_engine):
  """
  Create a new database session for each test function.
  Automatically rolls back after each test.
  """
  with Session(db_engine) as session:
    yield session
    session.rollback()


@pytest.fixture
def mock_logger():
  """Create a mock logger for testing."""
  logger = Mock()
  logger.info = Mock()
  logger.debug = Mock()
  logger.error = Mock()
  logger.warning = Mock()
  return logger


@pytest.fixture
def mock_dataset(db_session):
  """
  Create a sample dataset in database for testing.
  Uses random features and target from tagname.csv.
  """
  from app.database.orm import Dataset
  from app.database.schemas import TaskType, StatusProcess
  from tests.core.utils import get_random_features_target

  features, target = get_random_features_target(min_features=3, max_features=5)

  dataset = Dataset(
      name=f"test-dataset-{random.randint(1000, 9999)}",
      task_type=TaskType.Regression,
      features=features,
      target=target,
      status=StatusProcess.SUCCESS_PULL,
      is_valid=True,
      start_date="20240101",
      end_date="20240131",
      time_start="00:00:00",
      time_end="23:59:00",
      interval=5,
      description="Test regression dataset",
      preprocessing=None,
      top_model=None,
      meta={},
  )

  db_session.add(dataset)
  db_session.commit()
  db_session.refresh(dataset)

  return dataset


@pytest.fixture
def mock_classification_dataset(db_session):
  """Create a classification dataset for testing."""
  from app.database.orm import Dataset
  from app.database.schemas import TaskType, StatusProcess
  from tests.core.utils import get_random_features_target

  features, target = get_random_features_target(min_features=3, max_features=5)

  dataset = Dataset(
      name=f"test-classification-{random.randint(1000, 9999)}",
      task_type=TaskType.Classification,
      features=features,
      target=target,
      status=StatusProcess.SUCCESS_PULL,
      is_valid=True,
      start_date="20240101",
      end_date="20240131",
      time_start="00:00:00",
      time_end="23:59:00",
      interval=5,
      description="Test classification dataset",
      preprocessing=None,
      top_model=None,
      meta={},
  )

  db_session.add(dataset)
  db_session.commit()
  db_session.refresh(dataset)

  return dataset


@pytest.fixture
def mock_model_ml():
  """Create a mock ModelML object."""
  model = Mock()
  model.algorithm = "rf"
  model.name = "test_model"
  model.path = "/path/to/model.pkl"
  model.is_active = True
  return model


@pytest.fixture
def mock_pycaret_module():
  """Create a mock PyCaret module with all necessary methods."""
  mod = Mock()

  # Setup methods
  mod.setup = Mock(return_value=None)
  mod.create_model = Mock(return_value=Mock())
  mod.save_model = Mock(return_value=None)
  mod.load_model = Mock(return_value=Mock())
  mod.predict_model = Mock(return_value=Mock())
  mod.finalize_model = Mock(return_value=Mock())
  mod.compare_models = Mock(return_value=[Mock(), Mock(), Mock()])
  mod.pull = Mock(return_value=Mock())
  mod.get_config = Mock(return_value=Mock())

  return mod


@pytest.fixture
def mock_dataframe():
  """Create a mock pandas DataFrame for testing."""
  import pandas as pd

  df = Mock(spec=pd.DataFrame)
  df.columns = Mock()
  df.values = Mock()
  df.to_dict = Mock(return_value={"col1": [1, 2, 3]})
  return df


@pytest.fixture
def sample_features():
  """Return sample features for testing."""
  return {
      "feature1": [1.0, 2.0, 3.0],
      "feature2": [4.0, 5.0, 6.0],
  }


@pytest.fixture
def mock_config():
  """Mock Config class."""
  from pathlib import Path

  config = Mock()
  config.dir = Path("/tmp/test")
  config.window_size = 20
  config.debug_mode = False
  config.verbose = False
  return config
