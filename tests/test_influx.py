import warnings,random,pprint

from pandas._libs.tslibs import timestamps
from app.database.influx import InfluxDBStorage,InfluxDBClient
from influxdb_client.rest import ApiException
from influxdb_client.client.warnings import MissingPivotFunction
from app.config import Config
from app.helpers import DTEncoder
warnings.simplefilter("ignore", MissingPivotFunction)

URL = Config.influxdb_url
TOKEN = Config.influxdb_token
ORG = Config.influxdb_org
BUCKET = Config.influxdb_bucket
DATASET_NAME = "Regression-test"
N_WRITE = 10

class TestInflux:
  TEMP:dict = {}
  def test_auth_token(self):
    with InfluxDBStorage(url=URL,token=TOKEN,org=ORG) as influx:
      assert influx.check_auth(), "Unauthorized token!"
    influx.close()

  def test_buckets_is_available(self):
    with InfluxDBStorage(url=URL,token=TOKEN,org=ORG) as influx:
      buckets_api = influx.client.buckets_api()
      buckets = buckets_api.find_buckets()
      result = []
      for bucket in buckets.buckets:
        filter = bucket.name.startswith("_")
        if filter: result.append(bucket.name)
      assert result, "Buckets is Empty!"

  def test_delete_bucket(self):
    with InfluxDBStorage() as influx: print(influx.delete_bucket())

  def test_write_data(self):
    results = [{"Model_1":random.random(),"Model":random.random()} for _ in range(N_WRITE)]
    features = [{"Features_1":random.randint(3,100),
                 "Features_2":random.randint(3,100),
                 "Feature_3":random.randint(3,50)} for _ in range(N_WRITE)]
    dates = []
    with InfluxDBStorage() as influx:
      for idx,res in enumerate(results):
        now = DTEncoder.now_utc()
        status = influx.write_inference(dataset_name=DATASET_NAME,
                              task_type="Regression",results=res,
                              features=features[idx],actual=20.2,timestamp=now)
        dates.append(now)
        assert status, "Failed Write Data"
      self.TEMP["results"] = results
      self.TEMP["features"] = features
      self.TEMP["dates"] = dates
      pprint.pprint(self.TEMP)

  def test_query_data(self):
    with InfluxDBStorage() as influx:
      dfs = influx.query_inference(dataset_name=DATASET_NAME)
      assert len(dfs) == len(self.TEMP["results"]) == len(self.TEMP["features"])
      temp = self.TEMP
      for idx,data in enumerate(dfs):
        ts1 = data['timestamp']
        ts2 = temp['dates'][idx]
        print(ts1,"\t",ts2)

  # def test_query_data(self):
  #   query = f'''
  #   from(bucket: "{BUCKET}")
  #     |> range(start: 0,
  #         stop:  2100-01-01T00:00:00Z)
  #     |> limit(n: 10000)           // batasi jumlah data (ubah sesuai kebutuhan)
  #   '''
  #   with InfluxDBStorage(url=URL,token=TOKEN,org=ORG) as influx:
  #     dfs = influx.query_api.query_data_frame(query)
  #     n = 0
  #     for df in dfs:
  #       n += len(df)
  #       print(df["_time"])

  def test_delete_all_data(self):
    # self.test_write_data()
    with InfluxDBStorage() as influx:
      status = influx.delete_dataset("Regression-test")
      assert status, f"delete dataset is failed"

if __name__ == "__main__":
  TestInflux().test_delete_bucket()
  TestInflux().test_write_data()
  TestInflux().test_query_data()
  TestInflux().test_delete_bucket()
  # pytest.main([__file__, "-v", "-s"])
