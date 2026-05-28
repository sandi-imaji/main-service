from app.database.influx import InfluxDBStorage,write_inference
import datetime


def delete_dataset(dataset_name:str):
  with InfluxDBStorage() as writer:
    q = writer.delete_dataset(dataset_name=dataset_name)
    print(q)


def test():
  from datetime import datetime, timedelta

  a = "2026-02-27 23:55:01.001000+07:00"
  start_dt = datetime.fromisoformat(a)

  past_datetimes = [start_dt - timedelta(minutes=5) * i for i in range(1, 101)]

# Jika ingin dalam bentuk string ISO
  past_strings = [str(dt) for dt in past_datetimes]
  return past_strings[-1]


if __name__ == "__main__":
  import pprint
  # dataset_name = "Regression-97f7703b"
  # dataset_name = "Clustering-9ead5835"
  dataset_name = "TimeSeries-321a3acd"

  now = datetime.datetime.now()
  from app.helpers import DTEncoder
  from datetime import datetime
  # write_inference(dataset_name="Cluster-xyz",
  # task_type="unsupervised",
  # timestamp="2026-03-16T10:00:00",
  # results={"kmeans": 1, "birch": 0},
  # features={"f1": 25.4})
  with InfluxDBStorage() as writer:
    writer.delete_dataset(dataset_name)
    # query = writer.query_inference(dataset_name)
    # pprint.pprint(query)


    # print(query)
    # pprint.pprint(query,indent=2)

    


