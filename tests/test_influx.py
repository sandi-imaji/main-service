import pytest,pandas as pd
from unittest.mock import MagicMock
from app.database.influx import InfluxDBStorage,InfluxDBClient
from influxdb_client.rest import ApiException
from app.config import Config

URL = Config.influxdb_url
TOKEN = Config.influxdb_token
ORG = Config.influxdb_org
BUCKET = Config.influxdb_bucket

class TestInflux:
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

  def test_query_data(self):
    query = f'''
    from(bucket: "{BUCKET}")
      |> range(start: 0)           // dari awal waktu (semua data)
      |> limit(n: 10000)           // batasi jumlah data (ubah sesuai kebutuhan)
    '''
    with InfluxDBStorage(url=URL,token=TOKEN,org=ORG) as influx:
      table = influx.query(query)
      dfs = influx.query_api.query_data_frame(query)
      n = 0 
      for df in dfs:
        print(df.columns)
        n += len(df)
      print(n)

if __name__ == "__main__":
  TestInflux().test_query_data()
  # pytest.main([__file__, "-v", "-s"])
