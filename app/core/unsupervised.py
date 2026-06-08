"""
Unsupervised Learning Module (Clustering & Anomaly)
Optimized for performance with model caching and parallel processing support.
Phase 3: Async support dengan unified cache
"""
import sys
import pandas as pd
import numpy as np
import os
import datetime
import time
import re
import joblib
import json
import traceback
from typing import  Optional, Dict ,List
from contextlib import closing
from pathlib import Path

if __name__ == "__main__":
  # Get the directory containing this file: app/core/
  current_dir = Path(__file__).parent
  # Go up to parent: app/
  app_dir = current_dir.parent
  # Go up again to root project
  root_dir = app_dir.parent
  # Add root to sys.path if not already there
  if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.database.schemas import (
    TaskType,
    StatusProcess,
    MetaModel,
    MetaDataset,
)
from app.database.orm import Dataset, ModelML
from app.database.db import get_session
from app.database.schemas import PreprocessingSchema
from app.helpers import DTEncoder, pca
from sqlmodel import Session
# from app.database.influx import InfluxDBStorage
from app.config import Config
from app.pull import pull_realtime
from app.utils.model_cache import get_unsupervised_cache
from app.logger import Logger,LoggerNone
from app.core.base import BaseMLCore, InferenceMixin, ResultSchema
from uuid import uuid4
from pydantic import BaseModel
from pycaret import clustering as cl_mod


class UnsupervisedResultSchema(BaseModel, ResultSchema):
  features: Dict[str, int | float]
  features_reduce: Optional[Dict[str, int | float]] = None
  timestamp: datetime.datetime
  dataset_name: str
  clusters: Dict
  task_type: str = "Clustering"
  is_valid: bool = False

  # def write_to_influx(self, influx: InfluxDBStorage, logger):
  #   logger.info(f"[Efficient] Got InfluxDB storage: {influx.bucket}")
  #   timestamp = DTEncoder.to_utc(self.timestamp)
  #   write_result = influx.write_inference(
  #       dataset_name=self.dataset_name,
  #       task_type=self.task_type,
  #       timestamp=timestamp,
  #       results=self.clusters,
  #       features=self.features
  #   )
  #   logger.info(f"[Efficient] Write result: {write_result}")





# =============================================================================
# Model Cache - Now menggunakan unified cache dari app.utils.model_cache
# =============================================================================


def get_unsupervised_cache_compat():
  """Get the global unsupervised model cache instance (backward compatibility)."""
  return get_unsupervised_cache()


# =============================================================================
# PCA Helper
# =============================================================================

def _transform_pca(dataset_name: str, data: pd.DataFrame, logger) -> Optional[pd.DataFrame]:
  fpath = Config.dir/"storages"/dataset_name/"pca.joblib"
  logger.info("transform features to n-2 dimensionality")
  if not fpath:
    logger.warning(f"pca object is not found : {fpath}")
    return None
  pca_obj = joblib.load(fpath)
  data = pca_obj.transform(data)
  return pd.DataFrame(data, columns=["X1", "X2"])


# =============================================================================
# Module Loading
# =============================================================================


def module(task_type: TaskType):
  """
  Get the appropriate PyCaret module based on task type.
  Uses lazy loading for better startup performance.
  """
  if task_type.is_clustering():
    from pycaret import clustering

    return clustering
  elif task_type.is_anomaly():
    from pycaret import anomaly
    return anomaly
  else:
    raise ValueError(f"Task Type : {task_type} in Unsupervised")


# =============================================================================
# Unsupervised Learning Class
# =============================================================================


class Unsupervised(BaseMLCore, InferenceMixin):
  """
  Unsupervised learning operations with optimizations:
  - Model caching for faster repeated inference
  - Parallel model training support
  - Thread-safe operations
  """
  _task_type = "unsupervised"

  @classmethod
  def get_cache(cls):
    """Get the model cache for unsupervised learning."""
    return get_unsupervised_cache()

  @staticmethod
  def is_same_cluster(clusters:dict,name:Optional[str]= None) -> bool:
    values = set(clusters.values())
    result = len(values) == 1
    if not name: return result
    return result and name == list(values)[0]


  @staticmethod
  def auto_inference(dataset: Dataset, logger) -> UnsupervisedResultSchema:
    """
    Automated inference for clustering task.
    Fetches real-time data and runs cluster assignment.

    Args:
      dataset: Dataset object
      logger: Logger instance

    Returns:
      Dictionary with clustering results
      {'features': {'E2-TH-1-D-01-TEMP': 25.42, 'E2-TH-1-D-02-TEMP': 25.23, 'E2-TH-1-D-03-TEMP': 25.17},
      'features_reduce': {'X0': 5.515822710424747, 'X1': 0.35940568416911534},
      'dt': '2026-03-11 02:07:59.636575+07:00',
      'clusters': {'birch': 0.0, 'kmeans': 1.0}}

    """
    logger.info("auto inference start ...")
    try:
      features = dataset.features
      X = pull_realtime(features, logger)
      features = {col: [X[i]] for i, col in enumerate(features)}
      clusters = Unsupervised.inference(dataset, features, logger)
      return clusters

    except Exception as e: logger.error(str(e))

  @staticmethod
  def auto_inference_write_loop(dataset: Dataset, logger) -> Optional[Dict]:
    logger.info("start auto inference write loop ...")
    n = 1
    try:
      while True:
        logger.info(f"auto infernce - {n}")
        Unsupervised.auto_inference(dataset, logger)
        time.sleep(dataset.interval * 60)
    except Exception as e:
      logger.error(str(e))
      logger.error(traceback.format_exc())
      return None
    except KeyboardInterrupt:
      return None

  @staticmethod
  def inference(
      dataset: Dataset, features: Dict, logger=None
  ) -> UnsupervisedResultSchema:
    """
    Run inference with optional top-model-only mode.

    Args:
        dataset: Dataset object
        features: Dictionary of feature values
        only_top_model: If True, only use top model
        logger: Logger instance

    Returns:
        Dictionary with cluster assignments
    """
    mod = module(dataset.task_type)
    train_df = dataset.open_dataframe().iloc[1:]
    features = pd.DataFrame.from_dict(features)
    features["dt"] = DTEncoder.now()
    combined_df = pd.concat([train_df,features],ignore_index=True)
    result_df = combined_df.copy()
    n_clusters = int(dataset.target)

    clusters = {}
    if dataset.preprocessing:
      args = PreprocessingSchema(**dataset.preprocessing).to_args_pycaret()
      mod.setup(combined_df,ignore_features=["dt"],**args)
    else: mod.setup(combined_df,ignore_features=["dt"])

    naming_clusters = Unsupervised.get_naming_clusters(dataset.name)

    if logger: logger.info("Inference models : ")
    for m in dataset.models:
      algorithm = m.algorithm
      model = mod.create_model(algorithm,num_clusters=n_clusters,verbose=Config.verbose.pycaret)
      result = mod.assign_model(model,verbose=Config.verbose.pycaret)
      result_df[algorithm] = result["Cluster"]
      result["dt"] = combined_df["dt"]
      cluster = result.tail(1)["Cluster"].item()
      if naming_clusters: clusters[algorithm] = naming_clusters[algorithm][cluster]
      else: clusters[algorithm] = cluster

      if logger: logger.info(f"Inference with {m.algorithm}")
    features = features.drop(columns="dt")

    result = UnsupervisedResultSchema(features=features.iloc[0].to_dict(), dataset_name=dataset.name,
                                      clusters=clusters, timestamp=DTEncoder.now(), is_valid=True)

    if len(dataset.features) > 2: result.features_reduce = _transform_pca(dataset.name, features, logger).iloc[0].to_dict()
    fpath = Config.dir/"storages"/dataset.name
    combined_df.to_csv(fpath/"data.csv",index=False)

    result_df.to_csv(fpath/"results"/"clusters.csv",index=False)
    return result

  # @staticmethod
  # def inference_spec_model(dataset:Dataset,model_name:str,logger):pass

  @staticmethod
  def get_clusters(dataset: Dataset, logger) -> Dict:
    """
    Get cluster assignments for the dataset.

    Args:
      dataset: Dataset object
      logger: Logger instance

    Returns:
      Dictionary with cluster data
    """
    if not dataset.models or len(dataset.models) == 0: raise ValueError(f"Dataset {dataset.name} has no trained models yet")

    fpath = Config.dir / "storages" / dataset.name / "results/clusters.csv"
    if not os.path.exists(fpath): raise FileNotFoundError(f"clusters.csv not found for dataset {dataset.name}. Please ensure training is complete.")

    algorithms = [a.algorithm for a in dataset.models]
    feature_names = dataset.features

    result = pd.read_csv(fpath)

    naming_clusters = Unsupervised.get_naming_clusters(dataset.name)
    if naming_clusters:
      logger.info("renaming clusters ...")
      result = Unsupervised.rename_clusters(result,naming_clusters)

    result_dict = result.to_dict(orient="list")
    result_dict["feature_names"] = feature_names
    result_dict["algorithms"] = algorithms

    if len(feature_names) > 2:
      features = result[feature_names]
      feature_reduce = _transform_pca(dataset.name,features,logger=logger)
      if feature_reduce is None: return {}
      result_dict["X1"] = feature_reduce["X1"].values.tolist()
      result_dict["X2"] = feature_reduce["X2"].values.tolist()
    else:
      result_dict["X1"] = result_dict[feature_names[0]]
      result_dict["X2"] = result_dict[feature_names[1]]

    return result_dict

  @staticmethod
  def get_cluster_unique(dataset:Dataset):
    fpath = Config.dir/"storages"/dataset.name/"results"/"clusters.csv"
    if not fpath.exists(): raise ValueError(f"clusters.csv is not found in : {fpath}")
    df = pd.read_csv(fpath)
    algorithms = [a.algorithm for a in dataset.models]
    data =  df[algorithms].to_dict(orient="list")
    data = {key:list(set(data[key])) for key in data}
    return data

  @staticmethod
  def get_naming_clusters(dataset_name:str) -> Optional[Dict]:
    fpath = Config.dir/"storages"/dataset_name/"results"/"naming_clusters.json"
    if not fpath.exists(): return None
    with open(fpath,"r") as f: return json.load(f)

  @staticmethod
  def get_naming_cluster(dataset_name:str,algorithm:str,cluster_raw:str) -> Optional[str]:
    data = Unsupervised.get_naming_clusters(dataset_name)
    if not data:return None
    return data[algorithm][cluster_raw]


  @staticmethod
  def naming_clusters(clusters:pd.DataFrame,naming:Dict[str,str],logger):
    if not all(clust in naming.keys() for clust in  np.unique(clusters)):
      message = f"{np.unique(clusters).tolist()} not in naming : {list(naming.keys())}"
      logger.error(message)
      raise ValueError(message)
    return [naming[clust] for clust in clusters]

  @staticmethod
  def save_naming_clusters(dataset:Dataset,naming:Dict[str,Dict[str,str]]):
    fpath = Config.dir/"storages"/dataset.name/"results"/"naming_clusters.json"
    # fpath_clusters = Config.dir/"storages"/dataset.name/"results"/"clusters.csv"
    # dataset = pd.read_csv(fpath_clusters)
    # dataset = Unsupervised.rename_clusters(dataset,naming,inplace=False)
    # dataset.to_csv(fpath_clusters,index=False)
    with open(fpath,"w") as f: json.dump(naming,f)


  @staticmethod
  def update_naming_clusters(dataset:Dataset,update_naming:Dict[str,Dict[str,str]]):
    fpath = Config.dir/"storages"/dataset.name/"results"/"naming_clusters.json"
    if not fpath.exists():
      Unsupervised.save_naming_clusters(dataset,update_naming)
    else:
      naming_clusters = Unsupervised.get_naming_clusters(dataset.name)
      for model, clusters in naming_clusters.items():
        mapper = update_naming.get(model, {})
        for cluster_name, value in clusters.items():
          if value in mapper: clusters[cluster_name] = mapper[value]
      Unsupervised.save_naming_clusters(dataset,naming_clusters)
      return naming_clusters
  
  @staticmethod
  def rename_clusters(df: pd.DataFrame,
                    naming_clusters: dict,
                    inplace: bool = False) -> pd.DataFrame:
    """
    Rename cluster label berdasarkan mapping tiap model.

    Parameters
    ----------
    df : pd.DataFrame DataFrame cluster result.
    
    naming_clusters : dict
      Format:
      {
        "col_name": {
          cluster_id: cluster_name
        }
      }

    suffix : str
        Suffix untuk kolom hasil naming.

    inplace : bool
        Jika True maka modifikasi langsung dataframe asli.

    Returns
    -------
    pd.DataFrame
    """

    target = df if inplace else df.copy()

    for col, mapping in naming_clusters.items():
      if col not in target.columns: continue

      target[col] = target[col].map(mapping)
    return target


  @staticmethod
  def find_top_model(dataset:Dataset,n_top: int, logger,db:Session):
    """
    Find and train top N clustering models.

    Args:
        dataset_name: Name of the dataset
        n_top: Number of top models to select
        logger: Logger instance
    """

    try:
      task_type = dataset.task_type
      num_clusters = int(dataset.target)
      mod = module(task_type)

      df = dataset.open_dataframe()
      if len(dataset.features) > 2:
        pca_obj = pca(df[dataset.features].copy(),
                      n_components=2, only_obj=True)
        fpath = (
            Config.dir
            / "storages"
            / dataset.name
            / f"pca.joblib"
        )
        joblib.dump(pca_obj, fpath)
        logger.info("Saved PCA Object...")

      # Setup PyCaret
      if dataset.preprocessing:
        preprocessing = PreprocessingSchema(**dataset.preprocessing)
        args = preprocessing.to_args_pycaret()
        mod.setup(df, verbose=Config.verbose.pycaret,ignore_features=["dt"],**args)
      else:
        mod.setup(df, verbose=Config.verbose.pycaret,
                  ignore_features=["dt"])

      logger.info(f"Features : {mod.get_config('X').columns.tolist()}")

      algorithms = task_type.algorithms()
      models = []
      results = []
      times = {}

      # Train models sequentially (PyCaret doesn't support parallel training well)
      for algo, desc in algorithms.items():
        tic = time.monotonic()
        model = mod.create_model(
            algo, num_clusters=num_clusters, verbose=Config.verbose.pycaret
        )
        models.append(model)

        result = mod.pull().copy()
        result_dict = result.iloc[0].to_dict()
        result_dict["Algorithm"] = algo
        result_dict["Description"] = desc

        toc = time.monotonic()
        times[algo] = toc - tic
        results.append(result_dict)
        logger.info(f"{algo} successfully trained!")

      # Sort by metrics
      logger.info("Sorting Scores ...")
      ranked = sorted(
        results,
        key=lambda x: (
          -(
              x.get("Silhouette", -1e9)
              if x.get("Silhouette") is not None
              else -1e9
          ),
          -(
              x.get("Calinski-Harabasz", -1e9)
              if x.get("Calinski-Harabasz") is not None
              else -1e9
          ),
          (
              x.get("Davies-Bouldin", 1e9)
              if x.get("Davies-Bouldin") is not None
              else 1e9
          ),
        ),
      )

      top_model_name = ""
      for idx, m in enumerate(ranked[:n_top]):
        key = m["Algorithm"]
        path = f"storages/{dataset.name}/top_model/{key}"
        model_path = Config.dir / path
        mod.save_model(models[idx], str(model_path))

        df[key] = mod.assign_model(models[idx])["Cluster"]

        metric = {
            k: v
            for k, v in m.items()
            if k not in ["Algorithm", "Description"]
        }
        metric["TT (Sec)"] = times[key]

        model_name = f"{key}-{str(uuid4())[:8]}"

        meta = MetaModel(
            created_by="Anonymous",
            train_time=times[key],
            created_at=datetime.datetime.now().isoformat(),
            size_of=os.path.getsize(f"{model_path}.pkl"),
            notes="",
        )

        q_model = ModelML(
            name=model_name,
            algorithm=key,
            is_active=True,
            evaluation=metric,
            meta=meta.model_dump(),
            status=StatusProcess.SUCCESS_TRAIN,
            path=path,
        )
        dataset.models.append(q_model)

        if idx == 0:
          top_model_name = model_name

      # Save clusters
      fpath = Config.dir / "storages" / dataset.name / "results/clusters.csv"
      naming_clusters = Unsupervised.get_naming_clusters(dataset.name)
      if naming_clusters: Unsupervised.rename_clusters(df,naming_clusters,True)

      df.to_csv(fpath, index=False)

      # Update metadata
      current_meta = MetaDataset(**dataset.meta)
      total_train_times = sum(times.values())
      print(f"Total train times : {total_train_times}")
      current_meta.train_time = total_train_times
      dataset.meta = current_meta.model_dump()
      dataset.top_model = top_model_name
      dataset.status = StatusProcess.IDLE

      db.commit()
      logger.info("Compare Models Finished")

      # Clear cache after training
      get_unsupervised_cache().clear()

    except Exception as e:
      logger.error(f"Error in find_top_model: {str(e)}", exc_info=True)
      dataset.status = StatusProcess.ERROR_TRAIN
      db.commit()
      # db.rollback()

  @staticmethod
  def train(dataset: Dataset, algorithm: str, logger):
    ### untuk saat ini tidak digunakan, lihat find_top_model ###
    """
    Train a specific clustering algorithm.

    Args:
        dataset: Dataset object
        algorithm: Algorithm identifier
        logger: Logger instance
    """
    mod = module(dataset.task_type)
    df = dataset.open_dataframe()

    if dataset.preprocessing:
      args = PreprocessingSchema(**dataset.preprocessing).to_args_pycaret()
      mod.setup(df, verbose=Config.verbose.pycaret, ignore_features=["dt"],**args)
    else: mod.setup(df, verbose=Config.verbose.pycaret, ignore_features=["dt"])

    model = mod.create_model(
        algorithm, num_clusters=int(dataset.target), verbose=Config.verbose.pycaret
    )
    metric = mod.pull().to_dict(orient="index")[0]

    model_name = f"{algorithm}-{str(uuid4())[:8]}"
    path_model = f"storages/{dataset.name}/top_model/{algorithm}"
    mod.save_model(model, str(Config.dir / path_model))

    logger.info(f"Save Model at {Config.dir / path_model}")

    meta = MetaModel(
        created_by="Anonymous",
        created_at=datetime.datetime.now().isoformat(),
        train_time=0.0,
        size_of=os.path.getsize(f"{Config.dir}/{path_model}.pkl"),
        notes="",
    )

    q_model = ModelML(
        name=model_name,
        algorithm=algorithm,
        is_active=True,
        evaluation=metric,
        meta=meta.model_dump(),
        status=StatusProcess.SUCCESS_TRAIN,
        path=path_model,
    )

    dataset.models.append(q_model)
    logger.info(f"Model : {algorithm} Successfully")
    dataset.status = StatusProcess.IDLE
    logger.info(f"Train Model {algorithm} Finished!")

    # Clear cache
    get_unsupervised_cache().clear()


def main():
  if len(sys.argv) > 1:
    dataset_name = sys.argv[1]
    db = next(get_session())
    dataset = Dataset.get_by_name(dataset_name,db)
    print(dataset_name)
    if not dataset: raise ValueError("Dataset is not found!")
    logger = LoggerNone(dataset_name)
    Unsupervised.auto_inference_write_loop(dataset,logger=logger)
  else:
    print(sys.argv)
    print("Dataset name is None")

if __name__ == "__main__":
  # main()
  from app.logger import Logger
  dataset_name = "Clustering-1b4f986c"
  logger = Logger("test")
  db = next(get_session())
  dataset = Dataset.get_by_name(dataset_name, db)
  if not dataset: raise ValueError("Dataset is not found")

  # naming_clusters = {
  #   "sc": {"Cluster 0":"A","Cluster 1":"B","Cluster 2":"C"},
  #   "birch":{"Cluster 0":"AA","Cluster 1":"BB","Cluster 2":"CC"}
  # }
  #
  # _i = {key:{v:k for k,v in naming_clusters[key].items()} for key in naming_clusters}
  # # print(_i)
  #
  # get_unique : Dict[str,List] = {key:list(naming_clusters[key].values()) for key in naming_clusters}
  #
  # update_naming = {
  #   "sc": {"A":"AB","B":"BA","C":"BB"},
  #   "birch":{"AA":"A1","BB":"A2","CC":"A3"}
  # }
  #
  # data = Unsupervised.get_naming_clusters(dataset.name)
  print(Unsupervised.auto_inference(dataset,logger))


  # data = Unsupervised.get_naming_clusters(dataset.name)
  # print(data)
  # }
  # data = Unsupervised.update_naming_clusters(naming_clusters,update_naming)
  # print(data)







 

