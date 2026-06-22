"""
Supervised Learning Module (Classification & Regression)
Optimized for performance with model caching and batch inference support.
Phase 3: Async support dengan unified cache
"""

from typing import List, Optional, Dict
from contextlib import closing
from pathlib import Path
from uuid import uuid4
from pydantic import BaseModel
from datetime import datetime,timedelta
import os,pandas as pd,time,sys

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
from app.database.db import get_session,Session,get_db_session
from app.database.schemas import PreprocessingSchema
from app.helpers import DTEncoder, read_bytes_to_dict
from app.pull import pull_realtime,PullDate,pull_history
from app.database.influx import InfluxDBStorage, get_influx_storage
from app.utils.model_cache import get_supervised_cache
from app.config import Config
from app.core.base import BaseMLCore, InferenceMixin,ResultSchema
from app.logger import Logger,LoggerNone


# =============================================================================
# Model Cache - Now menggunakan unified cache dari app.utils.model_cache
# =============================================================================


def get_model_cache():
  """Get the global model cache instance (backward compatibility)."""
  return get_supervised_cache()


# =============================================================================
# Module Loading
# =============================================================================


def module(task_type: TaskType):
  """
  Get the appropriate PyCaret module based on task type.
  Uses lazy loading for better startup performance.
  """
  if task_type.is_classification():
    from pycaret import classification

    return classification
  elif task_type.is_regression():
    from pycaret import regression

    return regression
  else:
    raise ValueError(f"Task Type : {task_type} in Supervised")


def get_config(mod, logger):
  """Log feature names and target from PyCaret config."""
  feature_names = mod.get_config("X").columns.tolist()
  target_name = mod.get_config("target_param")
  logger.info(f"Features : {feature_names}")
  logger.info(f"Target : {target_name}")


class SupervisedResultSchema(BaseModel,ResultSchema):
  features: Dict[str,float|int]
  predictions: Dict[str,float|int]
  timestamp : datetime
  actual : Optional[int|float] = None
  task_type :str = "Regression"
  dataset_name:str
  is_valid:bool = False

  def write_to_influx(self,influx:InfluxDBStorage,logger):
    timestamp = DTEncoder.to_utc(self.timestamp)
    write_result = influx.write_inference(dataset_name=self.dataset_name,
        task_type=self.task_type,
        timestamp=timestamp,
        results=self.predictions,
        features=self.features,
        actual=self.actual
    )
    logger.info(f"[Efficient] Write result: {write_result}")

# =============================================================================
# Supervised Learning Class
# =============================================================================


class Supervised(BaseMLCore, InferenceMixin):
  """
  Supervised learning operations with optimizations:
  - Model caching for faster repeated inference
  - Batch inference support
  - Thread-safe operations
  """

  _task_type = "supervised"

  @classmethod
  def get_cache(cls):
    """Get the model cache for supervised learning."""
    return get_supervised_cache()

  @staticmethod
  def read_results(dataset: Dataset, logger) -> List[dict]:
    """
    Read historical prediction results from binary storage.

    Args:
        dataset: Dataset object
        logger: Logger instance

    Returns:
        List of result dictionaries
    """
    logger.info("Reading results ...")
    algorithms = [m.algorithm for m in dataset.models]
    fpath = Config.dir / "storages" / dataset.name / "results/results.bin"

    if not os.path.exists(fpath):
      logger.error(f"Reading error : FileNotFoundError")
      raise FileNotFoundError(f"results : {fpath} is not found!")

    data = read_bytes_to_dict(
        fpath=str(fpath), columns=["ts", "actual"] + algorithms
    )
    logger.info("reading success")
    return data

  @staticmethod
  def auto_inference(dataset: Dataset,logger) -> SupervisedResultSchema:
    """
    Automated inference: fetches features and actual value, then predicts.

    Args:
        dataset: Dataset object with features and target
        logger: Logger instance

    Returns: ResultSchema
    """
    logger.info("Auto Inference Start ...")
    try:
      features_names = dataset.features
      columns = features_names + [dataset.target]

      if Config.debug_mode:
        logger.info("Use Sampling from dataframe ...")
        df = dataset.open_dataframe()
        pulled = df.sample(1).drop(columns="dt")
        pulled_list = pulled[columns].values.flatten().tolist()
      else:
        pulled_list = pull_realtime(columns, logger)
      
      # NOTE: mungkin nanti bisa menggunakan logic checking 'all values is None'
      if sum(pulled_list) == 0:
        actual = 0
        algorithm = [m.algorithm for m in dataset.models]
        result = {k:0.0 for k in algorithm}
        features = {k:0.0 for k in dataset.features}
        return SupervisedResultSchema(features=features,dataset_name=dataset.name, predictions=result,
                                      timestamp=DTEncoder().now().isoformat(),
                                      actual=actual)

      X = pulled_list[:-1]
      actual = pulled_list[-1]

      # Build features dict efficiently
      features = {name: [X[i]] for i, name in enumerate(features_names)}

      # Run inference
      results = Supervised.inference(dataset, features=features, logger=logger)
      results.actual = actual

      logger.info("Auto Inference Finished!")
      return results

    except Exception as e:
      logger.error(f"Auto inference error: {str(e)}")

  @staticmethod
  def auto_inference_write_loop(dataset:Dataset,logger):
    logger.info("start auto inference write loop ...")
    try:
      while True:
        if dataset.is_time_to_finetune(): 
          logger.info("it's Time to finetune Dataset :) ")
          Supervised.finetune(dataset.name,logger)
        results = Supervised.auto_inference(dataset=dataset,logger=logger)
        if not results.is_valid: continue
        logger.info(f" Writing to InfluxDB...")
        try:
          influx = get_influx_storage()
          logger.info(f"[Efficient] Got InfluxDB storage: {influx.bucket}")
          results.write_to_influx(influx,logger)
        except Exception as e:
          logger.error(f"[Efficient] Failed to write to InfluxDB: {e}")
          logger.error(traceback.format_exc())
          # Don't raise - continue even if InfluxDB fails
        time.sleep(dataset.interval * 60)
    except Exception as e:
      logger.error(str(e))
      return None
    except KeyboardInterrupt: return None

  @staticmethod
  def inference(dataset: Dataset, features: dict, logger) -> SupervisedResultSchema:
    """
    Run inference on all models for given features.

    Optimized with model caching for repeated predictions.

    Args:
        dataset: Dataset object
        features: Dictionary of feature values {"x1": [value], "x2": [value]}
        logger: Logger instance

    Returns:
        SupervisedResultSchema Class : 
        {'features': {'E2-TH-1-D-01-TEMP': 25.42, 'E2-TH-1-D-02-TEMP': 25.23, 'E2-TH-1-D-03-TEMP': 25.17},
        'timestamp': '2026-03-11 02:07:59.636575+07:00',
        'predictions': {'knn': 20.19, 'lr': 20.20}}
    """
    try:
      mod = dataset.task_type.module()
      features = pd.DataFrame.from_dict(features)
      predictions = {}

      logger.info("Inference models: ")
      for m in dataset.models:
        logger.info(f"Inference with {m.algorithm}")
        modelpath = Config.dir / m.path

        # Use cached model loading
        model = Supervised._load_model_cached(mod, str(modelpath), logger)

        # Run prediction
        res = mod.predict_model(model, features)
        predictions[m.algorithm] = res.to_dict(orient="list")["prediction_label"][0]

      logger.info("Inference finished!")
      features = features.iloc[0].to_dict()
      return SupervisedResultSchema(features=features,predictions=predictions,
                                    timestamp=DTEncoder.now(),dataset_name=dataset.name,is_valid=True)

    except Exception as e:
      logger.error(str(e))
      predictions = {m.name:0.0 for m in dataset.models}
      return SupervisedResultSchema(features=features,predictions=predictions,timestamp=DTEncoder.now(),dataset_name=dataset.name)


  @staticmethod
  def find_top_model(dataset: Dataset, n_top: int, logger,db:Session):
    """
    Find and train top N models using PyCaret compare_models.

    Args:
        dataset_name: Name of the dataset
        n_top: Number of top models to select
        logger: Logger instance
    """
    n_top = 2
    try:
      tic = time.monotonic()
      task_type = dataset.task_type
      mod = module(task_type)
      df = dataset.open_dataframe()

      # Setup PyCaret
      if dataset.preprocessing:
        preprocessing = PreprocessingSchema(**dataset.preprocessing)
        args = preprocessing.to_args_pycaret()
        mod.setup(
            df,
            target=dataset.target,
            verbose=Config.verbose.pycaret,
            ignore_features=["dt"],
            n_jobs=1,
            **args
        )
      else:
        mod.setup(
            df,
            target=dataset.target,
            verbose=Config.verbose.pycaret,
            ignore_features=["dt"],
            n_jobs=1
        )
      get_config(mod, logger)

      # Compare and select top models
      top_model = mod.compare_models(n_select=n_top, verbose=Config.verbose.pycaret)
      metrics = mod.pull().to_dict("index")
      keys = list(metrics.keys())

      for i in range(n_top):
        model_name = f"{keys[i]}-{str(uuid4())[:8]}"
        evaluation = metrics[keys[i]]
        del evaluation["Model"]

        path_model = f"storages/{dataset.name}/top_model/{keys[i]}"
        model = mod.finalize_model(top_model[i])
        mod.save_model(model, str(Config.dir / path_model))

        logger.info(f"Save Model at {Config.dir / path_model}")

        meta = MetaModel(
            created_by="Anonymous",
            created_at=DTEncoder.now().isoformat(),
            train_time=0.0,
            size_of=os.path.getsize(f"{Config.dir}/{path_model}.pkl"),
            notes="",
        )

        q_model = ModelML(
            name=model_name,
            algorithm=keys[i],
            is_active=True,
            evaluation=evaluation,
            meta=meta.model_dump(),
            status=StatusProcess.SUCCESS_TRAIN,
            path=path_model,
        )
        dataset.models.append(q_model)
        logger.info(f"Model : {keys[i]} Successfully !")

      dataset.top_model = dataset.models[0].name
      dataset.status = StatusProcess.IDLE

      toc = time.monotonic()
      meta = MetaDataset(**dataset.meta)
      meta.train_time = toc - tic
      dataset.meta = meta.model_dump()

      db.commit()
      logger.info("Compare Models Finished!")

      # Clear model cache after training new models
      get_model_cache().clear()

    except Exception as e:
      logger.error(str(e))
      dataset.status = StatusProcess.ERROR_TRAIN
      db.commit()
      db.rollback()

  @staticmethod
  def finetune(dataset_name:str,logger):
    logger.info("start finetune ...")
    try:
      with get_db_session() as db:
        dataset = Dataset.get_by_name(dataset_name,db)
        if not dataset: raise ValueError("Dataset is not found!")
        status_before = dataset.status
        dataset.status = StatusProcess.RUNNING_FINETUNE
        logger.debug(f"Update Dataset Status to : {StatusProcess.RUNNING_FINETUNE}")
        db.commit()
        
        now = DTEncoder.now()
        if dataset.meta.get('current_dt',False): current_dt = datetime.fromisoformat(dataset.meta['current_dt'])
        else: current_dt = now

        current_dt = current_dt + timedelta(minutes=dataset.interval)
        if dataset.preprocessing:
          interval_finetune = dataset.preprocessing['interval_finetune']
        else: interval_finetune = 2

        n_out = interval_finetune // 2
        df_before = dataset.open_dataframe()

        columns = dataset.meta['columns']
        columns.remove('dt')
        start_dt_pull = PullDate.from_dt(current_dt)
        end_dt_pull = PullDate.from_dt(now)
        end_dt_pull.time_start = "00:00:00"
        
        if dataset.preprocessing: preprocessing = PreprocessingSchema(**dataset.preprocessing)
        else: preprocessing = PreprocessingSchema()

        df_new = pull_history(columns=columns,start_date=start_dt_pull,
                                end_date=end_dt_pull,logger=logger,preprocessing=preprocessing)
        all_data = pd.concat([df_before,df_new]).sort_values(by="dt").reset_index(drop=True)


        # # Truncate
        start_date = DTEncoder.str_to_dt(dataset.start_date) + timedelta(days=n_out)
        filtered_df = all_data[(all_data['dt'] >= pd.Timestamp(start_date.date(), tz='UTC'))\
                                & (all_data['dt'] <= pd.Timestamp(now.date(), tz='UTC'))].drop_duplicates(subset=["dt"])

        # # Train 
        mod = dataset.task_type.module()

        cache = get_model_cache()
        preprocessing_args = preprocessing.to_args_pycaret()
        mod.setup(filtered_df,target=dataset.target,verbose=Config.verbose.pycaret,ignore_features=["dt"],
                  **preprocessing_args)

        for m in dataset.models:
          model = mod.create_model(m.algorithm,verbose=Config.verbose.pycaret)
          metric = mod.pull().loc["Mean"].to_dict()
          model = mod.finalize_model(model)

          path_model = f"storages/{dataset.name}/top_model/{m.algorithm}"

          if os.path.exists(Config.dir / path_model):
            os.remove(Config.dir / path_model)
            logger.info(f"Delete Model Previous : {m.algorithm}")
            cache.invalidate(str(Config.dir / path_model))

          mod.save_model(model, str(Config.dir / path_model))
          logger.info(
              f"Save Model Update: {m.algorithm} at {Config.dir / path_model}"
          )

          model_meta = MetaModel(
              created_by="Anonymous",
              last_update=datetime.now().isoformat(),
              created_at=m.meta.get("created_at", "")
              if isinstance(m.meta, dict)
              else getattr(m.meta, "created_at", ""),
              size_of=os.path.getsize(f"{Config.dir}/{path_model}.pkl"),
              notes=f"Update {datetime.now().isoformat()}",
          )

          m.evaluation = metric
          m.meta = model_meta.model_dump()
          db.commit()
          logger.info(f"Model Update : {m.algorithm} Successfully")

        dataset.start_date = DTEncoder.dt_to_str(start_date)
        dataset.end_date = DTEncoder.dt_to_str(now)

        dataset_meta = MetaDataset(**dataset.meta)
        print(dataset_meta)
        dataset_meta.current_dt = now.isoformat()
        dataset_meta.last_update = now.isoformat()
        dataset.meta = dataset_meta.model_dump()
        dataset.status = status_before
        db.commit()

        filtered_df.to_csv(str(Config.dir/"storages"/dataset.name/"data.csv"),index=False)
        logger.info(f"Dataset {dataset.name} Successfully Finetune!")
        get_supervised_cache().clear()

    except Exception as e: logger.error(f"Error Finetune : {e}")


  @staticmethod
  def train(dataset: Dataset, algorithm: str, logger):
    """
    Train a specific algorithm on the dataset.

    Args:
        dataset: Dataset object
        algorithm: Algorithm identifier to train
        logger: Logger instance
    """
    mod = module(dataset.task_type)
    df = dataset.open_dataframe()
    
    if dataset.preprocessing:
      args = PreprocessingSchema(**dataset.preprocessing).to_args_pycaret()
      mod.setup( df, target=dataset.target, verbose=Config.verbose.pycaret, ignore_features=["dt"],**args)
    else: mod.setup( df, target=dataset.target, verbose=Config.verbose.pycaret, ignore_features=["dt"])
    get_config(mod, logger)

    model = mod.create_model(algorithm, verbose=Config.verbose.pycaret)
    metric = mod.pull().loc["Mean"].to_dict()
    model_name = f"{algorithm}-{str(uuid4())[:8]}"
    path_model = f"storages/{dataset.name}/top_model/{algorithm}"

    model = mod.finalize_model(model)
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

    # Clear cache for this model path if it exists
    get_model_cache().clear()

def main():
  if len(sys.argv) > 1:
    dataset_name = sys.argv[1]
    db = next(get_session())
    dataset = Dataset.get_by_name(dataset_name,db)
    print(dataset_name)
    if not dataset: raise ValueError("Dataset is not found!")
    logger = LoggerNone(dataset_name)
    Supervised.auto_inference_write_loop(dataset,logger=logger)
  else:
    print(sys.argv)
    print("Dataset name is None")


if __name__ == "__main__":
  main()
