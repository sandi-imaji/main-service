from app.routes.modelML import auto_initialize
from app.database.db import get_session
from app.database.orm import Dataset


db = next(get_session())

dataset_name = "Regression-443f7225"

data = Dataset.get_by_name(dataset_name,db)

if not data: raise ValueError("Dataset is not found!")

auto_initialize(data,3)
