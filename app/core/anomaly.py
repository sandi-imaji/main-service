"""
Anomaly Detection Module
Optimized for performance with lazy loading and model caching.
Phase 3: Async support dengan unified cache
"""

from dataclasses import is_dataclass
import pandas as pd,datetime,time,traceback,sys,json
from pathlib import Path
from typing import Dict
from pydantic import BaseModel

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


from pycaret import anomaly as mod
from app.database.db import get_session
from app.database.orm import Dataset
from app.database.schemas import TaskType
from app.database.influx import InfluxDBStorage, get_influx_storage
from app.pull import pull_realtime
from app.utils.model_cache import get_anomaly_cache
from app.logger import Logger, LoggerNone
from app.config import Config
from app.helpers import DTEncoder
from app.core.base import BaseMLCore, InferenceMixin,ResultSchema
# from app.core.unsupervised import Unsupervised


# =============================================================================
# Anomaly Detection Class
# =============================================================================

class AnomalyResultSchema(BaseModel,ResultSchema):
  timestamp:datetime.datetime
  features:Dict[str,int|float]
  is_anomaly:bool
  anomaly_score:int|float
  dataset_name:str
  is_valid:bool = False

  def write_to_influx(self,influx:InfluxDBStorage,logger) -> None:
    logger.info(f"[Efficient] Got InfluxDB storage: {influx.bucket}")
    timestamp = DTEncoder.to_utc(self.timestamp)
    dataset_name = f"{self.dataset_name}"
    write_result = influx.write_inference(
      dataset_name= dataset_name,
      task_type=self.task_type,
      timestamp=timestamp,
      results={"is_anomaly":self.is_anomaly,"anomaly_score":self.anomaly_score},
      features=self.features
    )
    logger.info(f"[Efficient] Write result: {write_result}")

  @property
  def task_type(self): return "Anomaly"
  
  @staticmethod
  def invalid(dataset_name:str):
    features = {"":0.0}
    return AnomalyResultSchema(timestamp=DTEncoder.now(),
                               features=features,dataset_name=dataset_name,
                               is_anomaly=False,anomaly_score=0.0,is_valid=False)


class Anomaly(BaseMLCore, InferenceMixin):
  """
  Anomaly detection operations with optimizations:
  - Lazy model loading (only when needed)
  - Model caching for faster repeated inference
  - Thread-safe operations
  """

  ALGORITHM = "iforest"  # Default algorithm: Isolation Forest
  _task_type = "anomaly"

  @classmethod
  def get_cache(cls):
    """Get the model cache for anomaly detection."""
    return get_anomaly_cache()

  @staticmethod
  def auto_inference(dataset:Dataset,logger):
    logger.info("Auto Inference Start ...")
    columns = dataset.meta.get("columns","")
    if not columns: columns = dataset.open_dataframe().drop(columns='dt').columns
    if "dt" in columns: columns.remove("dt")
    pulled = pull_realtime(columns)
    if not pulled: pulled = [0.0 for _ in columns]
    features = {val:[pulled[idx]] for idx,val in enumerate(columns)}
    result = Anomaly.inference(dataset,features,logger)
    return result

  @staticmethod
  def auto_inference_write_loop(dataset:Dataset,logger):
    logger.info("start auto inference write loop ...")
    idx = 1
    try:
      while True:
        logger.info(f"auto inference - {idx}")
        results = Anomaly.anomaly(dataset=dataset,logger=logger)
        if results.is_valid:
          logger.info(f" Writing to InfluxDB...")
          try:
            influx = get_influx_storage()
            results.write_to_influx(influx,logger)
          except Exception as e:
            logger.error(f"[Efficient] Failed to write to InfluxDB: {e}")
            logger.error(traceback.format_exc())
            # Don't raise - continue even if InfluxDB fails
        time.sleep(dataset.interval * 60)
        idx += 1
    except Exception as e:
      logger.error(str(e))
      return None
    except KeyboardInterrupt: return None

  @staticmethod
  def inference(dataset:Dataset,features:Dict,logger) -> AnomalyResultSchema:
    """
    """
    try:
      mod = TaskType.Anomaly.module()
      features = pd.DataFrame.from_dict(features)

      logger.info(f"Inference Anomaly with {Anomaly.ALGORITHM}")
      modelpath = Config.dir / "storages"/dataset.name/"anomaly"
      # Use cached model loading
      model = Anomaly._load_model_cached(mod, str(modelpath), logger)
      # Run prediction
      res = mod.predict_model(model, features)[["Anomaly","Anomaly_Score"]].iloc[0].to_dict()

      logger.info("Inference finished!")
      features = features.iloc[0].to_dict()
      return AnomalyResultSchema(features=features,is_anomaly=bool(res['Anomaly']),anomaly_score=res["Anomaly_Score"],
                                 timestamp=DTEncoder.now(),dataset_name=dataset.name,is_valid=True)
    except Exception as e:
      logger.warning(str(e))
      return AnomalyResultSchema(features={},is_anomaly=False,
                                 anomaly_score=0.0,timestamp=DTEncoder.now(),dataset_name="",is_valid=False)

  @staticmethod
  def train(dataset: Dataset,algorithm:str,fraction:float, logger):
    """
    Train anomaly detection model using Isolation Forest.

    Args:
        dataset: Dataset object
        logger: Logger instance
    """
    logger.info("Training Dataset Anomaly ...")
    df = dataset.open_dataframe()

    if "dt" in df.columns.tolist(): df = df.drop(columns="dt")

    fpath = Config.dir / "storages" / dataset.name

    # Setup and train
    mod.setup(df, verbose=Config.verbose.pycaret)
    model = mod.create_model(algorithm,fraction=fraction)

    # Assign labels and save results
    df_with_labels = mod.assign_model(model)
    df_with_labels.to_csv(f"{fpath}/anomaly.csv", index=False)

    # Save model
    mod.save_model(model, f"{fpath}/anomaly")

    # Invalidate cache for this model
    get_anomaly_cache().invalidate(f"{fpath}/anomaly")

    logger.info("Successfully Trained Anomaly Model")
    return df_with_labels
  
  @staticmethod
  def anomaly(dataset:Dataset,logger):
    result = Anomaly.auto_inference(dataset,logger)
    logger.info(f"is_anomaly : {result.is_anomaly} | anomaly_score : {result.anomaly_score}")
    core = dataset.task_type.core()
    features = {key:[result.features[key]] for key in result.features}
    # fpath = Config.dir/"storages"/dataset.name/'cluster_anomaly.json'
    if dataset.task_type.is_clustering():
      clusters = core.inference(dataset,features,logger=logger)
      logger.info(f"Clusters : {clusters.clusters}")
      logger.info("Verify anomaly with cluster ...")
      if result.is_anomaly:
        is_same_clusters = core.is_same_cluster(clusters.clusters,"anomaly")
        if is_same_clusters: logger.info("Cluster is anomaly")
        result.is_anomaly = is_same_clusters
    return result

def main():
  if len(sys.argv) > 1:
    dataset_name = sys.argv[1]
    db = next(get_session())
    dataset = Dataset.get_by_name(dataset_name,db)
    if not dataset: raise ValueError("Dataset is not found!")
    logger = LoggerNone(dataset_name)
    Anomaly.auto_inference_write_loop(dataset,logger=logger)
  else:
    print(sys.argv)
    print("Dataset name is None")


if __name__ == "__main__":
  # from app.logger import Logger
  # import pprint,time
  # dataset_name = "Clustering-1b4f986c"
  # db = next(get_session())
  # logger = Logger("test")
  # dataset = Dataset.get_by_name(dataset_name,db)
  # if not dataset: raise ValueError("Dataset is not found!")
  # # data = Anomaly.train(dataset,algorithm="iforest",fraction=0.4,logger=logger)
  # result = Anomaly.anomaly(dataset,logger)
  # print(result)
  # Anomaly.clear_cache()
  main()
  
