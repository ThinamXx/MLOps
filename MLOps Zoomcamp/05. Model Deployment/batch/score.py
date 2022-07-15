# IMPORTING MODULES:
import os
import sys
import pickle 
import uuid
from numpy import save
import pandas as pd
from datetime import datetime
from prefect import get_run_logger, task, flow 
from prefect.context import get_run_context
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from dateutil.relativedelta import relativedelta

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


# FUNCTION TO SAVING RESULTS:
def save_results(df: pd.DataFrame, y_pred, run_id, output_file):
    df_result = pd.DataFrame()
    df_result["ride_id"] = df["ride_id"]
    df_result["lpep_pickup_datetime"] = df["lpep_pickup_datetime"]
    df_result["PULocationID"] = df["PULocationID"]
    df_result["DOLocationID"] = df["DOLocationID"]
    df_result["actual_duration"] = df["duration"]
    df_result["predicted_duration"] = y_pred
    df_result["diff"] = df_result["actual_duration"] - df_result["predicted_duration"]
    df_result["model_version"] = run_id
    
    df_result.to_parquet(output_file, index=False)



# IMPLEMENTATION OF THE MODEL:
@task
def apply_model(input_file, run_id, output_file):
    logger = get_run_logger()

    logger.info("Reading data from {}".format(input_file))
    df = read_dataframe(input_file)
    dicts = prepare_dictionaries(df)
    
    logger.info("Loading model with {}".format(run_id))
    model = load_model(run_id)

    logger.info("Applying model with {}".format(run_id))
    y_pred = model.predict(dicts)

    logger.info("Saving predictions to {}".format(output_file))
    save_results(df, y_pred, run_id, output_file)
    return output_file
    


# FUNCTION TO GET THE PATHS:
def get_paths(run_date, taxi_type, run_id):
    prev_month = run_date - relativedelta(months=1)
    year = prev_month.year  
    month = prev_month.month

    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/{taxi_type}-{year:04d}-{month:02d}.parquet'

    return input_file, output_file



@flow 
def ride_duration_prediction(
        taxi_type: str, 
        run_id: str,
        run_date: datetime=None):

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("green-taxi-duration")

    if run_date is None:
        ctx = get_run_context()
        run_date = ctx.flow_run.expected_start_time
    
    input_file, output_file = get_paths(run_date, taxi_type, run_id)

    apply_model(
        input_file=input_file, 
        run_id=run_id, 
        output_file=output_file
    )



# IMPLEMENTATION:
def run():
    taxi_type = sys.argv[1]  # "green"
    year = int(sys.argv[2])  # 2021
    month = int(sys.argv[3]) #  1
    run_id = sys.argv[4]     # "895c76c3ee6746cb96328b189042f646"

    ride_duration_prediction(
        taxi_type=taxi_type, 
        run_id=run_id, 
        run_date=datetime(year=year, month=month, day=1)
    )

if __name__ == "__main__":
    run()