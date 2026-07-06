import pytest,requests,datetime,pprint,json,time,random
from app.pull import query_tagnames,PullDate
from app.helpers import DTEncoder
from app.database.schemas import (DatasetRequestSchema,InitiateRequestSchema,
PreprocessingSchema,TaskType,StatusProcess,AnomalyRequestSchema)
from typing import Tuple

IP = "http://127.0.0.1:8004"
ENDPOINT = f"{IP}/models/auto-initiate"
N_DAYS = 3
N_MODELS = 2

def wait_until_finished(dataset_name:str,timeout=30,interval=0.5) -> Tuple[bool,StatusProcess]:
  start = time.monotonic()
  status = StatusProcess.RUNNING_PULL
  while time.monotonic() - start < timeout:
    resp = requests.get(f"{IP}/datasets/{dataset_name}/status")
    resp.raise_for_status()
    data = json.loads(resp.text)
    status = StatusProcess(data['status_code'])
    if status == StatusProcess.IDLE: return True,status
    time.sleep(interval)
  return status == StatusProcess.IDLE,status


class TestAutoInitiate:
  @pytest.mark.timeout(100)
  def test_supervised(self):
    features = [
      "CRAH-2DH2.1-SUPPLY_AIR_TEMP",
      "CRAH-2DH2.2-SUPPLY_AIR_TEMP",
      "CRAH-2DH2.3-SUPPLY_AIR_TEMP",
    ]
    target = "CRAH-2DH2.1-RETURN_AIR_TEMP"
    end_date = DTEncoder.now()
    start_date = end_date - datetime.timedelta(days=N_DAYS)
    description = "From Pytest"
    payload = InitiateRequestSchema(
      description=description,
      task_type = TaskType.Regression.name,
      features=features,target=target,start_date=DTEncoder.dt_to_str(start_date),
      end_date = DTEncoder.dt_to_str(end_date),time_start = "00:00:00",time_end="23:59:00",
      n_models=N_MODELS,preprocessing=PreprocessingSchema()
    )
    resp = requests.post(ENDPOINT,json=payload.model_dump(mode="json"))
    assert resp.status_code == 200, f"detail : {resp.status_code}"
    resp_data = json.loads(resp.text)
    assert resp_data.get("names",False),f"keys : {resp_data.keys()}"
    result,status = wait_until_finished(resp_data["names"],timeout=100,interval=3)
    assert result ,f"Status : {status}"

  @pytest.mark.timeout(100)
  def test_unsupervised(self):
    features = [
      "CRAH-2DH2.1-SUPPLY_AIR_TEMP",
      "CRAH-2DH2.2-SUPPLY_AIR_TEMP",
      "CRAH-2DH2.3-SUPPLY_AIR_TEMP",
      "CRAH-2DH2.4-SUPPLY_AIR_TEMP",
      "CRAH-2DH2.5-SUPPLY_AIR_TEMP",
    ]
    target = 3
    end_date = DTEncoder.now()
    start_date = end_date - datetime.timedelta(days=N_DAYS)
    description = "From Pytest"
    payload = InitiateRequestSchema(
      description=description,
      task_type = TaskType.Clustering.name,
      features=features,target=target,start_date=DTEncoder.dt_to_str(start_date),
      end_date = DTEncoder.dt_to_str(end_date),time_start = "00:00:00",time_end="23:59:00",
      n_models=N_MODELS,preprocessing=PreprocessingSchema()
    )
    resp = requests.post(ENDPOINT,json=payload.model_dump(mode="json"))
    assert resp.status_code == 200, f"detail : {resp.status_code}"
    resp_data = json.loads(resp.text)
    assert resp_data.get("names",False),f"keys : {resp_data.keys()}"
    result,status = wait_until_finished(resp_data["names"],timeout=100,interval=3)
    assert result ,f"Status : {status}"

  @pytest.mark.timeout(110)
  def test_timeseries(self):
    features = [
      "CRAH-2DH2.1-SUPPLY_AIR_TEMP",
    ]
    target = 20
    end_date = DTEncoder.now()
    start_date = end_date - datetime.timedelta(days=7)
    description = "From Pytest"
    payload = InitiateRequestSchema(
      description=description,
      task_type = TaskType.TimeSeries.name,
      features=features,target=target,start_date=DTEncoder.dt_to_str(start_date),
      end_date = DTEncoder.dt_to_str(end_date),time_start = "00:00:00",time_end="23:59:00",
      n_models=N_MODELS,preprocessing=PreprocessingSchema()
    )
    resp = requests.post(ENDPOINT,json=payload.model_dump(mode="json"))
    assert resp.status_code == 200, f"detail : {resp.status_code}"
    resp_data = json.loads(resp.text)
    assert resp_data.get("names",False),f"keys : {resp_data.keys()}"
    result,status = wait_until_finished(resp_data["names"],timeout=120,interval=3)
    assert result ,f"Status : {status}"

  @pytest.mark.timeout(110)
  def test_anomaly(self):
    task_type = "Regression"
    resp = requests.get(f"{IP}/datasets/filter/{task_type}")
    resp.raise_for_status()
    data = json.loads(resp.text)["data"]
    if not data: raise ValueError("Dataset is not found!")
    dataset_name = random.choice([d["name"] for d in data])

    list_algorithms = ("abod","cluster","cof","histogram","iforest","knn","lof","svm",
                 "pca","mcd","sod","sos")

    algorithm = random.choice(list_algorithms)
    
    fractions = random.uniform(0.1,0.5)

    payload = AnomalyRequestSchema(dataset_name=dataset_name,algorithm=algorithm,fraction=fractions)

    resp = requests.post(f"{IP}/models/anomaly/auto",json=payload.model_dump(mode="json"))
    assert resp.status_code == 200, f"{resp.text}"

if __name__ == "__main__":
  pytest.main([__file__, "-v", "-s"])
  TestAutoInitiate().test_supervised()
  # TestAutoInitiate.test_supervised()

