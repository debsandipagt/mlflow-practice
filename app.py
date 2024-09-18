# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature




def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha")
    parser.add_argument("--l1-ratio")
    args = parser.parse_args()

    # Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
    wine_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wine-quality.csv")
    data = pd.read_csv(wine_path)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = float(args.alpha)
    l1_ratio = float(args.l1_ratio)

    remote_server_uri = "https://dagshub.com/debsandipagt/mlflow.mlflow"
    mlflow.set_tracking_uri(remote_server_uri)

    # import dagshub
    # dagshub.init(repo_owner='debsandipagt', repo_name='mlflow', mlflow=True)

    with mlflow.start_run():
        mlflow.log_artifact(wine_path)
        #data = pd.read_csv(wine_path)

        # Log dataset metadata (optional, e.g., dataset name, size)
        dataset_size = os.path.getsize(wine_path)  # Get dataset size in bytes
        mlflow.log_param("dataset_name", "wine-quality.csv")
        mlflow.log_param("dataset_size_bytes", dataset_size)

        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)
        signature = infer_signature(train_x, predicted_qualities)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print(f"Elasticnet model (alpha={alpha:f}, l1_ratio={l1_ratio:f}):")
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        
        

        tracking_url_parse_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_parse_store != "file":
            mlflow.sklearn.log_model(
                lr, "model", registered_model_name="ElesticNetWineModel", signature=signature
            )

        else:
            mlflow.sklearn.log_model(lr, "model", signature=signature)