from app.core.contracts import ClusterRequest, ClusteringTrainRequest
from app.core.unsupervised import Unsupervised
from app.database.schemas import PreprocessingSchema
from app.config import Config
from app.logger import Logger
import pandas as pd


logger = Logger("test")

df = pd.read_csv("tests/data.csv").sample(1000)
features = [c for c in df.columns if c != "dt"]
X_sample = df.sample(1)

ps = PreprocessingSchema(dim_reduce=True, scale=True)

out_dir = Config.dir/"tests"
payload_train = ClusteringTrainRequest(df=df,preprocessing=ps.to_args_pycaret(),out_dir=out_dir,
                                       n_top=2,n_clusters=2,task="Clustering")

Unsupervised.compare_models(payload_train,Logger("test"))

# Transductive predict: the query point(s) must be appended to the training
# rows — predict() reads the LAST row of req.df as the point being assigned.
# combined_df = pd.concat([df, X_sample], ignore_index=True)
#
# algorithms = ["birch", "kmeans", "sc"]
#
# payload_preds = ClusterRequest(
#     df=combined_df,
#     algorithms=algorithms,
#     n_clusters=2,
#     preprocessing=ps.to_args_pycaret(),
#     task="Clustering",
# )
# print(Unsupervised.predict(payload_preds, logger=logger))
