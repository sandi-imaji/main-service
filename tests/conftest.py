"""
Root-level shared fixtures for tests outside tests/core/
(workers, orm, routes). Mirrors the in-memory SQLite pattern used by
tests/core/conftest.py so Dataset/ModelML tests never touch the real
sqlite file at storages/rtdb/data.db.
"""

import pytest
from sqlmodel import SQLModel, create_engine, Session
from sqlalchemy.pool import StaticPool
from unittest.mock import Mock

TEST_DATABASE_URL = "sqlite:///:memory:"


@pytest.fixture(scope="module")
def db_engine():
  """Fresh in-memory database engine, isolated per test module."""
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
  """New session per test function, rolled back afterwards."""
  with Session(db_engine) as session:
    yield session
    session.rollback()


@pytest.fixture
def mock_logger():
  """Mock logger compatible with app.logger.Logger's interface."""
  logger = Mock()
  logger.info = Mock()
  logger.debug = Mock()
  logger.error = Mock()
  logger.warning = Mock()
  return logger


@pytest.fixture
def client(tmp_path):
  """FastAPI TestClient with the DB dependency overridden to an isolated
  file-based SQLite database, so route tests never hit the real sqlite
  file or require a live server.

  Uses a file (not sqlite:///:memory:) because TestClient runs the ASGI
  app in a separate worker thread via an anyio portal; an in-memory
  StaticPool connection created in the main thread is not reliably
  visible from that worker thread, which manifests as spurious
  "no such table" errors.
  """
  from fastapi.testclient import TestClient
  from app.server import app
  from app.database.db import get_session

  db_path = tmp_path / "test.db"
  engine = create_engine(
      f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
  )
  SQLModel.metadata.create_all(engine)

  def override_get_session():
    with Session(engine) as session:
      yield session

  app.dependency_overrides[get_session] = override_get_session
  with TestClient(app) as test_client:
    yield test_client
  app.dependency_overrides.clear()
  SQLModel.metadata.drop_all(engine)
