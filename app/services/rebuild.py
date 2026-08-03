"""User-triggered dataset rebuilds.

Two operations exposed on the project page, both returning the dataset to a
clean state before training:

    rebuild  — drop workers + artifacts, RE-PULL the data, then find_top_models
    retrain  — drop workers + artifacts, go straight to find_top_models

Pulling is the only difference. Rebuild is for when the data itself needs
refreshing; retrain for when the data is fine but the models should be searched
again.

Workers are deliberately NOT restarted afterwards. Bringing a worker back up on
models that were swapped underneath it is a reliable source of confusing state;
the user presses Start Worker once they have seen the result.
"""
from __future__ import annotations

import os
import shutil

from app.config import Config
from app.database.base import get_db_session
from app.database.DB import Dataset
from app.database.schemas import StatusProcess
from app.exceptions import InvalidStateException
from app.logger import Logger
from app.pull import pulling
from app.services import train as train_service
from app.workers.manager import WorkerManager


def reset(dataset: Dataset, db, logger) -> dict:
  """Return a dataset to its pre-training state: no workers, no models.

  Deliberately NOT using `model_service.clean_models`: that function opens with
  `check_dataset_pretrained`, which raises when the dataset has never been
  trained or when a model file is missing. Broken state like that is the single
  most common reason someone reaches for Rebuild — the cleanup has to tolerate
  it rather than fail alongside it.

  Every worker of the dataset goes, the Anomaly one included: its model is not
  touched here, but leaving it running while the primary task's models are
  replaced mixes two generations of model in the same logs and results.
  """
  WorkerManager.delete_by_dataset(dataset.name)
  logger.info(f"Rebuild: workers for {dataset.name} deleted")

  model_count = len(dataset.models)
  for m in list(dataset.models):
    db.delete(m)
  dataset.top_model = ""
  db.commit()

  root = Config.dir / "storages" / dataset.name
  for folder in ("top_model", "results"):
    path = root / folder
    if os.path.exists(path):
      shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)

  logger.info(f"Rebuild: cleaned {model_count} models and their artifacts")
  return {"models_removed": model_count}


def _n_models(dataset: Dataset, n_models: int | None) -> int:
  """How many models to search for. A dataset may carry n_models=0 (created
  deliberately without training), but an explicit rebuild/retrain request always
  means "train" — so never fewer than one."""
  return max(1, n_models if n_models else dataset.n_models)


def rebuild(dataset_name: str, n_models: int | None = None) -> None:
  """Full rebuild: clean up, re-pull the data, then search for the best models."""
  logger = Logger(dataset_name)
  with get_db_session() as db:
    dataset = Dataset.get_by_name(dataset_name, db)
    n_top = _n_models(dataset, n_models)
    logger.info(f"REBUILD started | n_models: {n_top}")
    reset(dataset, db, logger)

    pulling(dataset_name, db)

    # `pulling` writes through its own session; re-read so the status and
    # is_valid seen here come from the pull that just happened, not a stale copy.
    dataset = Dataset.get_by_name(dataset_name, db)
    db.refresh(dataset)
    if not dataset.is_valid or dataset.status != StatusProcess.SUCCESS_PULL:
      logger.error(f"REBUILD aborted: pull failed (status {dataset.status})")
      return

    train_service.find_top_models(dataset, n_top, db, logger)
    logger.info("REBUILD finished")


def retrain(dataset_name: str, n_models: int | None = None) -> None:
  """Retrain on the data already on disk — no pulling."""
  logger = Logger(dataset_name)
  with get_db_session() as db:
    dataset = Dataset.get_by_name(dataset_name, db)
    if not dataset.is_valid:
      raise InvalidStateException("Dataset", "Not Valid", "Valid")

    n_top = _n_models(dataset, n_models)
    logger.info(f"RETRAIN started | n_models: {n_top}")
    reset(dataset, db, logger)

    # `reset` removed the models, so the dataset is back to "pulled but not yet
    # trained" — exactly the precondition find_top_models works from.
    dataset.status = StatusProcess.SUCCESS_PULL
    db.commit()

    train_service.find_top_models(dataset, n_top, db, logger)
    logger.info("RETRAIN finished")
