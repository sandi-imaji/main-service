import pandas as pd,pprint,json,os
from sklearn.ensemble import IsolationForest
from collections import deque

def detect_outliers_isolation_forest(df):
  numeric_df = df.select_dtypes(include=['float64','int64'])
  
  model = IsolationForest(contamination=0.05)
  df['outlier'] = model.fit_predict(numeric_df)

  outliers = df[df['outlier'] == -1]
  return outliers

def detect_outliers_all_columns(df):
  numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
  
  outlier_indices = []

  for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)].index
    outlier_indices.extend(outliers)

  return df.loc[set(outlier_indices)]

def detect_outliers_iqr(df, column):
  Q1 = df[column].quantile(0.25)
  Q3 = df[column].quantile(0.75)
  IQR = Q3 - Q1

  lower_bound = Q1 - 1.8 * IQR
  upper_bound = Q3 + 1.8 * IQR

  outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
  return outliers


def write_data(data: dict):
  fpath = "./data_predictions.json"
  # load data lama
  if os.path.exists(fpath):
    try:
      with open(fpath, "r") as f: old_data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError): old_data = []
  else: old_data = []

  # FIFO max 20
  dq = deque(old_data, maxlen=300)
  dq.append(data)

  # simpan kembali
  with open(fpath, "w") as f: json.dump(list(dq), f, indent=2)
  print(f"Total data: {len(dq)}")

if __name__ == "__main__":
  df = pd.read_csv("examples_outlier.csv").sample(100).to_dict(orient="records")
  # tagname = "E2-TH-1-D-03-TEMP"
  # result = detect_outliers_iqr(df,tagname)
  # write_data(df.to_dict(orient="records"))
  # data  = df.to_dict(orient="records")
  for i in df: write_data(i)
  with open("data_predictions.json","r") as f:
    data = json.load(f)
    pprint.pprint(data)
  # pprint.pprint(data)
  

