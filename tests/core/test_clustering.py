import pytest,pandas as pd,joblib
from app.config import Config
from app.core.unsupervised import Unsupervised,module,_build_dataframe_pca
from app.pull import pull_realtime
from app.database.db import get_session
from app.database.orm import Dataset
from app.logger import Logger

db = next(get_session())

dataset_name = "Clustering-e11716c7"
logger = Logger("test")
dataset = Dataset.get_by_name(dataset_name,db)
if not dataset: raise ValueError("Dataset is not found!")

x = pull_realtime(dataset.features,logger=logger)
data = {dataset.features[idx]:[x[idx]]for idx in range(0,len(x))}
data = pd.DataFrame.from_dict(data)


# pca_path = Config.dir/"storages"/dataset.name/"pca.joblib"
# pca = joblib.load(pca_path)
# data_transform = pca.transform([x])[0]
# # data_transform = {["X1","X2"][idx]: [data_transform[idx]] for idx in range(2)}
# # df = pd.DataFrame.from_dict(data_transform)
# print(data_transform)


# df = pd.DataFrame.from_dict(data)
# print(df)


mod = module(dataset.task_type)
models = dataset.models

# print(_build_dataframe_pca(df=data,dataset_name=dataset_name,logger=logger))


# for m in models:
#     model = mod.load_model(m.path)
#     print(mod.predict_model(model,df))


print(Unsupervised.auto_inference(dataset,logger=logger))



