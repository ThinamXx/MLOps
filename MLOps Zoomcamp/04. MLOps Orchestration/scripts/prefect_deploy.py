# IMPORTING MODULES:
import pickle 
import pandas as pd 

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error

import mlflow
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner

#@ IGNORING WARNINGS: 
import warnings
warnings.filterwarnings("ignore")


# OVERALL FUNCTION TO PREPARE DATAFRAME:
@task
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
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']

    df[categorical] = df[categorical].astype(str)                                   # Conversion.
    return df

@task 
def add_features(df_train, df_val):
    # INITIALIZING DATAFRAME:
    # df_train = read_dataframe(train_path)                                           # Training dataframe.
    # df_val = read_dataframe(valid_path)                                             # Validation dataframe.

    # PROCESSING THE DATA:
    df_train["PU_DO"] = df_train['PULocationID'] + '_' + df_train['DOLocationID']
    df_val["PU_DO"] = df_val["PULocationID"] + "_" + df_val["DOLocationID"]

    # INITIALIZING ONE HOT ENCODING:
    categorical = ["PU_DO"]
    numerical = ["trip_distance"] 

    dv = DictVectorizer()                                                       # Initializing dict vectorizer.
    train_dicts = df_train[categorical + numerical].to_dict(orient="records")   # Initializing dictionary.
    X_train = dv.fit_transform(train_dicts)                                     # Initializing one hot encoding.

    val_dicts = df_val[categorical + numerical].to_dict(orient="records")       # Initializing dictionary.
    X_val = dv.transform(val_dicts)                                             # Initializing one hot encoding.

    target = "duration"
    y_train = df_train[target].values                                           # Initialization.
    y_val = df_val[target].values

    return X_train, X_val, y_train, y_val, dv

@task
def train_model_search(train, valid, y_val):

    # OBJECTIVE FUNCTION TO TRACK PARAMETERS: XGBOOST:
    def objective(params):
        with mlflow.start_run():                  # Initializing mlflow tracker.
            mlflow.set_tag("model", "xgboost")    # Tracking model.
            mlflow.log_params(params)             # Tracking parameters.
            
            booster = xgb.train(
                params=params,
                dtrain=train, 
                num_boost_round=100,
                evals=[(valid, "validation")],
                early_stopping_rounds=50
            )                                      # Initializing xgb training.
            
            y_pred = booster.predict(valid)        # Generating predictions.
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)        # Tracking metric.
        
        return {"loss": rmse, "status": STATUS_OK}

    # INITIALIZING SEARCH SPACE:
    search_space = {
        "max_depth": scope.int(hp.quniform("max_depth", 4, 100, 1)),
        "learning_rate": hp.loguniform("learning_rate", -3, 0),
        "reg_alpha": hp.loguniform("reg_alpha", -5, -1),
        "reg_lambda": hp.loguniform("reg_lambda", -6, -1),
        "min_child_weight": hp.loguniform("min_child_weight", -1, 3),
        "objective": 'reg:linear',
        "seed": 22
    }

    # TRAINING AND OPTIMIZING XGBOOST:
    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=1,
        trials=Trials()
    )

    return 

@task
def train_best_model(train, valid, y_val, dv):
    with mlflow.start_run():
        
        params = {
            "learning_rate": 0.11228477728991716,
            "max_depth": 1, 
            "min_child_weight": 1.8357365049718906, 
            "objective": 'reg:linear',
            "reg_alpha": 0.012071686964333202,
            "reg_lambda": 0.3087106233718785,
            "seed": 2022
        }                                             # Initializing best parameters.
        
        mlflow.log_params(params)                     # Logging parameters.
        
        booster = xgb.train(
                params=params,
                dtrain=train, 
                num_boost_round=100,
                evals=[(valid, "validation")],
                early_stopping_rounds=50
            )                                          # Initializing xgboost training.
        
        y_pred = booster.predict(valid)                # Generating predictions.
        rmse = mean_squared_error(y_val, y_pred, 
                                squared=False)
        
        mlflow.log_metric("rmse", rmse)                # Tracking metric.
        
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        
        mlflow.log_artifact("models/preprocessor.b",
                        artifact_path="preprocessor")                      # Tracking dict vectorizer.
        
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")   # Tracking model artifacts. 


@flow(task_runner=SequentialTaskRunner)
def main(train_path: str="./data/green_tripdata_2021-01.parquet", 
         valid_path: str="./data/green_tripdata_2021-02.parquet"):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc-taxi-exp")
    X_train = read_dataframe(train_path)
    X_val = read_dataframe(valid_path)
    X_train, X_val, y_train, y_val, dv = add_features(X_train, X_val).result()
    train = xgb.DMatrix(X_train, label=y_train)             # Training dataset.
    valid = xgb.DMatrix(X_val, label=y_val)                 # Validation dataset.
    train_model_search(train, valid, y_val)                 # Training model.
    train_best_model(train, valid, y_val, dv)               # Training best model.


# INITIALIZING DEPLOYMENT:
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import IntervalSchedule
from prefect.flow_runners import SubprocessFlowRunner
from datetime import timedelta


# DEFINING DEPLOYMENT SPEC:
DeploymentSpec(
    flow=main,
    name="model_training",
    schedule=IntervalSchedule(interval=timedelta(minutes=5)),
    flow_runner=SubprocessFlowRunner(),
    tags=["ml"]
)
