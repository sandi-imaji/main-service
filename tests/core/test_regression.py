import pytest
from app.core.supervised import Supervised
from app.database.db import get_session
from app.database.orm import Dataset
from app.logger import Logger

db = next(get_session())

logger = Logger("test")
dataset_name = "Regression-eab29644"
dataset = Dataset.get_by_name(dataset_name,db)

print(Supervised.auto_inference(dataset,logger))

