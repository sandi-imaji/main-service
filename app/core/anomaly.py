"""
Anomaly Detection Module (pure ML core).

Compute only — no Dataset / DB / Influx. Training pulls its inputs from an
AnomalyTrainRequest; inference from a PredictRequest. The service layer
(app.services) owns pulling live data, persistence, and the InfluxDB write side.
"""
import pandas as pd

from app.utils.model_cache import get_anomaly_cache
from app.config import Config
from app.core.base import BaseMLCore
from app.core.contracts import AnomalyTrainRequest, PredictRequest, TrainedModel, AnomalyResult
from pycaret import anomaly as mod


class Anomaly(BaseMLCore):
  """
  Anomaly detection operations with optimizations:
  - Lazy model loading (only when needed)
  - Model caching for faster repeated inference
  """

  ALGORITHM = "iforest"  # Default algorithm: Isolation Forest
  _task_type = "anomaly"

  @classmethod
  def get_cache(cls):
    """Get the model cache for anomaly detection."""
    return get_anomaly_cache()

  @staticmethod
  def predict(req: PredictRequest, logger) -> AnomalyResult:
    """
    Run anomaly inference. Pure compute: loads the (single) saved model via the
    shared cache and returns is_anomaly + score. Raises on failure; the caller
    decides how to handle it.
    """
    features_df = pd.DataFrame.from_dict(req.features)
    _, path = req.models[0]                     # anomaly keeps a single model

    logger.info(f"Inference Anomaly with {Anomaly.ALGORITHM}")
    model = Anomaly._load_model_cached(mod, path, logger)
    res = mod.predict_model(model, features_df)[["Anomaly", "Anomaly_Score"]].iloc[0].to_dict()
    logger.info("Inference finished!")

    return AnomalyResult(features=features_df.iloc[0].to_dict(),is_anomaly=bool(res["Anomaly"]), anomaly_score=res["Anomaly_Score"])

  @staticmethod
  def train_one(req: AnomalyTrainRequest, logger) -> TrainedModel:
    """
    Train an anomaly-detection model. Pure compute (no Dataset/DB): saves the
    model and the labelled CSV under the dataset storage root, and returns its
    descriptor.

    Anomaly keeps a single model at `<storage>/anomaly` (not per-algorithm in
    top_model/), so the paths are derived from `req.out_dir.parent`.
    """
    logger.info("Training Dataset Anomaly ...")
    df = req.df
    algorithm = req.algorithm
    if "dt" in df.columns.tolist(): df = df.drop(columns="dt")
    storage_root = req.out_dir.parent          # out_dir == <storage>/top_model

    mod.setup(df, verbose=Config.verbose.pycaret, **req.preprocessing)
    model = mod.create_model(algorithm, fraction=req.fraction)

    # Assign labels and save results for later review.
    labelled = mod.assign_model(model)
    labelled.to_csv(storage_root / "anomaly.csv", index=False)

    path = storage_root / "anomaly"
    mod.save_model(model, str(path))
    get_anomaly_cache().invalidate(str(path))

    evaluation = Anomaly._summarise(labelled, req.fraction)
    logger.info(f"Successfully Trained Anomaly Model | {evaluation}")
    return TrainedModel.from_saved(algorithm, path, evaluation)

  @staticmethod
  def _summarise(labelled: pd.DataFrame, fraction: float) -> dict:
    """Summarise the detection result into numbers a user can judge.

    Anomaly has no comparison metric like the other tasks — there are no ground
    truth labels to score against, so no MAE or Silhouette can be computed. The
    consequence used to be an empty `{}` evaluation: no way to compare `iforest`
    against `knn`, no way to tell whether the chosen `fraction` was sensible, and
    the dashboard had to skip it entirely.

    What is offered here is purely descriptive, taken from columns
    `assign_model()` already produces:

    - `AnomalyRate` / `AnomalyCount` — how many were flagged. Compared against
      `FractionRequested`, this is how you see whether the model actually
      honoured what was asked of it.
    - `ScoreMean` / `ScoreP95` / `ScoreMax` — the score spread. The gap between
      P95 and Max shows whether anomalies stand out sharply or tail off; if they
      tail off, the threshold is essentially arbitrary.
    - `Threshold` — the lowest score still counted as an anomaly, i.e. the
      dividing line the model settled on.
    """
    anomaly = labelled["Anomaly"].astype(bool)
    score = labelled["Anomaly_Score"].astype(float)
    total = int(len(labelled))
    count = int(anomaly.sum())

    summary = {
        "AnomalyCount": count,
        "AnomalyRate": count / total if total else None,
        "FractionRequested": float(fraction),
        "TotalRows": total,
        "ScoreMean": float(score.mean()) if total else None,
        "ScoreP95": float(score.quantile(0.95)) if total else None,
        "ScoreMax": float(score.max()) if total else None,
    }
    # A threshold only means anything when something was actually flagged.
    summary["Threshold"] = float(score[anomaly].min()) if count else None
    return summary
