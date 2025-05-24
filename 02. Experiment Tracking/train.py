import os
import pickle
import click
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    mlflow.set_experiment("experiment-for-homework-2")
    mlflow.sklearn.autolog()

    with mlflow.start_run():
        mlflow.set_tag("developer", "Sujal Neupane")
        mlflow.log_param("train_data_path", "./data/green_trip_2023-01.csv")
        mlflow.log_param("valid_data_path", "./data/green_trip_2023-02.csv")

        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        max_depth = 10
        mlflow.log_param("max_depth", max_depth)
        rf = RandomForestRegressor(max_depth=max_depth, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mlflow.log_metric("rmse", rmse)
        
        print(f"RMSE: {rmse}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")

if __name__ == '__main__':
    run_train()