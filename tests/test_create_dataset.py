"""
Live-integration script (like tests/test_initiate.py): needs the server
running at localhost:8004. Queries tagnames matching a pattern, uses the
target tag as target and the rest as features, then auto-initiates a
Regression dataset. Run directly: PYTHONPATH=. python tests/test_create_dataset.py
"""
import requests,pprint,json,time,datetime
from typing import List

from app.helpers import DTEncoder
from app.database.schemas import InitiateRequestSchema,PreprocessingSchema,TaskType

IP = "http://localhost:8004"
N_DAYS = 3
N_MODELS = 2

query = "CRAH-2DH2.1-"

def get_query_names(q:str) -> List[str]:
  url = f"{IP}/datasets/utils/tagname?query={q}"
  response = requests.get(url)
  data = json.loads(response.text)
  return data

def create_datasets(tagnames:List[str]):
  target = "CRAH-2DH2.1-RETURN_AIR_TEMP"
  if target in tagnames: tagnames.remove(target)
  features = tagnames
  if not features: raise ValueError("No feature tagnames found from query!")

  end_date = DTEncoder.now()
  start_date = end_date - datetime.timedelta(days=N_DAYS)
  payload = InitiateRequestSchema(
    description="From test_create_dataset",
    task_type=TaskType.Regression.name,
    features=features,target=target,
    start_date=DTEncoder.dt_to_str(start_date),
    end_date=DTEncoder.dt_to_str(end_date),
    time_start="00:00:00",time_end="23:59:00",
    n_models=N_MODELS,preprocessing=PreprocessingSchema()
  )

  url = f"{IP}/models/auto-initiate"
  response = requests.post(url,json=payload.model_dump(mode="json"))
  response.raise_for_status()
  data = json.loads(response.text)
  pprint.pprint(data)
  return data



if __name__ == "__main__":
  tagnames = get_query_names("CRAH-2DH2.1-*")
  create_datasets(tagnames)
