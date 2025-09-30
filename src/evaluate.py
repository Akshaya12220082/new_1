import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow
from urllib.parse import urlparse
import warnings


os.environ['MLFLOW_TRACKING_URI']="https://dagshub.com/akshayavardhini2004/mlpipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME']="akshayavardhini2004"
os.environ["MLFLOW_TRACKING_PASSWORD"]="c96e7c595233e90b16f17d3267b098cf95598023"


# Load parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))["train"]

def evaluate(data_path,model_path):
    data=pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    mlflow.set_tracking_uri("https://dagshub.com/akshayavardhini2004/mlpipeline.mlflow")

    ## load the model from the disk
    model=pickle.load(open(model_path,'rb'))

    predictions=model.predict(X)
    accuracy=accuracy_score(y,predictions)
    ## log metrics to MLFLOW

    mlflow.log_metric("accuracy",accuracy)
    print(f"Model accuracy: {accuracy:.4f}")   # f-string


if __name__=="__main__":
    evaluate(params["data"],params["model"])



warnings.filterwarnings("ignore", category=UserWarning, module="mlflow.protos.service_pb2")
