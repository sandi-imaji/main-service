"""Shared dataset helpers for the service layer."""
from app.config import Config
from app.database.DB import Dataset
from app.exceptions import ValidationException
from app.utils.security import safe_path_join


def split_data(payload):
  """Split a dataset into train/test CSVs (with secure path handling)."""
  dataset: Dataset = payload.dataset
  dataset.check_integrity()

  df = dataset.get_df()
  if dataset.target not in df.columns:
    raise ValidationException("Target column not found", field=dataset.target)

  train = df.sample(frac=payload.train_size, random_state=42)
  test = df.drop(train.index)

  base_path = safe_path_join(Config.dir, "storages", dataset.name)
  train.to_csv(base_path / "train.csv", index=False)
  test.to_csv(base_path / "test.csv", index=False)
  return {"train": len(train), "test": len(test)}
