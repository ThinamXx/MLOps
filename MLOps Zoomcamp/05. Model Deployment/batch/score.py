# IMPORTING MODULES:
from distutils.util import run_2to3
import os
import sys
import pickle 
import uuid
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

#@ IGNORING WARNINGS: 
import warnings
warnings.filterwarnings("ignore")


# IMPORTING AND SETTING UP MLFLOW:
import mlflow
from mlflow.tracking import MlflowClient


# GENERATING UID:
def generate_uuids(n):
    ride_ids = []
    for i in range(n):
        ride_ids.append(str(uuid.uuid4()))
    return ride_ids

# OVERALL FUNCTION TO PREPARE DATAFRAME:
def read_dataframe(filename):                                                           # Defining function.
    if filename.endswith(".csv"):                                                       # Checking.
        df = pd.read_csv(filename)                                                      # Reading the dataset.
        df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)             # Converting to datetime. 
        df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)               # Converting to datetime. 
    elif filename.endswith(".parquet"):
        df = pd.read_parquet(filename)
        
    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda x: x.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    
    df["ride_id"] = generate_uuids(len(df))
    return df

def prepare_dictionaries(df: pd.DataFrame):
    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)
    
    df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]
    categorical = ["PU_DO"]
    numerical = ["trip_distance"] 
    dicts = df[categorical + numerical].to_dict(orient="records")   # Initializing dictionary.
    return dicts


# FUNCTION TO LOAD MODEL:
def load_model(run_id):
    logged_model = f'runs:/{run_id}/model'
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    return loaded_model



# IMPLEMENTATION OF THE MODEL:
def apply_model(input_file, run_id, output_file):
    print("Reading data from {}".format(input_file))
    df = read_dataframe(input_file)
    dicts = prepare_dictionaries(df)
    
    print("Loading model with {}".format(run_id))
    model = load_model(run_id)

    print("Applying model with {}".format(run_id))
    y_pred = model.predict(dicts)
    
    print("Saving predictions to {}".format(output_file))
    df_result = pd.DataFrame()
    df_result["ride_id"] = df["ride_id"]
    df_result["lpep_pickup_datetime"] = df["lpep_pickup_datetime"]
    df_result["PULocationID"] = df["PULocationID"]
    df_result["DOLocationID"] = df["DOLocationID"]
    df_result["actual_duration"] = df["duration"]
    df_result["predicted_duration"] = y_pred
    df_result["diff"] = df_result["actual_duration"] - df_result["predicted_duration"]
    df_result["model_version"] = run_id
    
    return df_result.to_parquet(output_file, index=False)



# IMPLEMENTATION:
def run():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("green-taxi-duration")

    taxi_type = sys.argv[1]  # "green"
    year = int(sys.argv[2])  # 2021
    month = int(sys.argv[3]) #  1

    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/{taxi_type}-{year:04d}-{month:02d}.parquet'

    run_id = sys.argv[4]     # "895c76c3ee6746cb96328b189042f646"


    apply_model(
        input_file=input_file, 
        run_id=run_id, 
        output_file=output_file
    )


if __name__ == "__main__":
    run()