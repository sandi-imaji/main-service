"""
Test utilities for loading tagnames and generating test data.
Uses tests/tagname.csv as source for features and targets.
"""

import csv
import random
from typing import List, Tuple, Optional
from pathlib import Path


def load_tagnames(csv_path: Optional[str] = None) -> List[str]:
  """
  Load all tagnames from CSV file.

  Args:
      csv_path: Path to CSV file. Defaults to tests/tagname.csv

  Returns:
      List of tagname strings
  """
  if csv_path is None:
    # Default path from project root
    csv_path = Path(__file__).parent.parent / "tagname.csv"

  with open(csv_path, "r") as f:
    reader = csv.DictReader(f)
    return [row["tagname"] for row in reader]


def get_random_features_target(
    min_features: int = 3, max_features: int = 10, seed: Optional[int] = None
) -> Tuple[List[str], str]:
  """
  Randomly select features and target from tagnames.

  Args:
      min_features: Minimum number of features to select
      max_features: Maximum number of features to select
      seed: Random seed for reproducibility (optional)

  Returns:
      Tuple of (features_list, target_string)
  """
  if seed is not None:
    random.seed(seed)

  tagnames = load_tagnames()

  # Random number of features
  n_features = random.randint(
      min_features, min(max_features, len(tagnames) - 1))

  # Random selection without replacement
  selected = random.sample(tagnames, n_features + 1)  # +1 for target

  features = selected[:-1]
  target = selected[-1]

  return features, target


def get_task_type_features(
    task_type: str,
    min_features: int = 3,
    max_features: int = 10,
    seed: Optional[int] = None,
) -> Tuple[List[str], str]:
  """
  Get features and target suitable for specific task type.

  Args:
      task_type: One of ['Classification', 'Regression', 'Clustering',
                         'TimeSeries', 'Anomaly']
      min_features: Minimum number of features
      max_features: Maximum number of features
      seed: Random seed for reproducibility

  Returns:
      Tuple of (features_list, target_string)
  """
  features, target = get_random_features_target(
      min_features, max_features, seed)

  if task_type in ["Clustering", "Anomaly"]:
    # For clustering/anomaly, target is number of clusters
    target = str(random.randint(2, 5))
  elif task_type == "TimeSeries":
    # For time series, use single feature with datetime target
    features = features[:1]  # Single feature
    target = "dt"  # Datetime column

  return features, target


def get_sample_data(
    n_samples: int = 100, n_features: int = 5, seed: Optional[int] = None
) -> Tuple[List[str], List[List[float]]]:
  """
  Generate sample feature names and random data.

  Args:
      n_samples: Number of samples (rows)
      n_features: Number of features (columns)
      seed: Random seed for reproducibility

  Returns:
      Tuple of (feature_names, data_matrix)
  """
  if seed is not None:
    random.seed(seed)

  # Get feature names
  tagnames = load_tagnames()
  feature_names = random.sample(tagnames, n_features)

  # Generate random data
  data = [
      [random.uniform(0, 100) for _ in range(n_features)] for _ in range(n_samples)
  ]

  return feature_names, data


def create_mock_dataframe(
    features: List[str], n_samples: int = 10, seed: Optional[int] = None
):
  """
  Create a mock pandas DataFrame for testing.

  Args:
      features: List of feature names
      n_samples: Number of rows
      seed: Random seed for reproducibility

  Returns:
      Mock DataFrame object
  """
  import pandas as pd
  from unittest.mock import Mock

  if seed is not None:
    random.seed(seed)

  # Create actual DataFrame
  data = {
      feature: [random.uniform(0, 100) for _ in range(n_samples)]
      for feature in features
  }
  df = pd.DataFrame(data)

  return df


def create_features_dict(
    features: List[str], n_samples: int = 1, seed: Optional[int] = None
) -> dict:
  """
  Create features dictionary for inference.

  Args:
      features: List of feature names
      n_samples: Number of samples
      seed: Random seed

  Returns:
      Dictionary with feature names as keys and lists of values
  """
  if seed is not None:
    random.seed(seed)

  return {
      feature: [random.uniform(0, 100) for _ in range(n_samples)]
      for feature in features
  }


# Cache loaded tagnames to avoid reading file multiple times
_tagnames_cache: Optional[List[str]] = None


def get_cached_tagnames() -> List[str]:
  """Get cached tagnames (loads once)."""
  global _tagnames_cache
  if _tagnames_cache is None:
    _tagnames_cache = load_tagnames()
  return _tagnames_cache
