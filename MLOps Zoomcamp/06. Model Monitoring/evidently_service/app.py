"""
DEMO SERVICE FOR EVIDENTLY METRICS INTEGRATION WITH PROMETHEUS AND GRAFANA:
"""

# IMPORTING MODULES:
from ast import Str
from curses import window
import os
import yaml
import hashlib
import dataclasses
import datetime
import logging
import flask
import pandas as pd
import prometheus_client

from flask import Flask
from pyarrow import parquet as pq
from typing import Dict, List, Optional
from werkzeug.middleware.dispatcher import DispatcherMiddleware

from evidently.runner.loader import DataLoader
from evidently.runner.loader import DataOptions
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.model_monitoring import ModelMonitoring, CatTargetDriftMonitor
from evidently.model_monitoring import ClassificationPerformanceMonitor, DataDriftMonitor
from evidently.model_monitoring import DataQualityMonitor, NumTargetDriftMonitor
from evidently.model_monitoring import ProbClassificationPerformanceMonitor, RegressionPerformanceMonitor


# INITIALIZING FLASK:
app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s) [%(levelname)s] %(message)s', handlers=[logging.StreamHandler()]
)


# ADDING PROMETHEUS WSGI MIDDLEWARE TO ROUTE /METRICS REQUESTS:
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {"/metrics": prometheus_client.make_wsgi_app()})


@dataclasses.dataclass
class MonitoringServiceOptions:
    datasets_path: str
    min_reference_size: str
    use_reference: bool
    moving_reference: bool
    window_size: int
    calculation_period_sec: int


@dataclasses.dataclass
class LoadedDataset:
    name: str 
    references: pd.DataFrame
    monitors: List[str]
    column_mapping: ColumnMapping


# INITIALIZING EVIDENTLY MAPPING:
EVIDENTLY_MONITORS_MAPPING = {
    "cat_target_drift": CatTargetDriftMonitor,
    "data_drift": DataDriftMonitor,
    "data_quality": DataQualityMonitor,
    "num_target_drift": NumTargetDriftMonitor,
    "regression_performance": RegressionPerformanceMonitor,
    "classification_performance": ClassificationPerformanceMonitor,
    "prob_classification_performance": ProbClassificationPerformanceMonitor
}


class MonitoringService:
    datasets: List[str]                                 # name of monitoring datasets.
    metric: Dict[str, prometheus_client.Gauge]
    last_run: Optional[datetime.datetime]
    reference: Dict[str, pd.DataFrame]                  # collection of reference data. 
    current: Dict[str, Optional[pd.DataFrame]]          # collection of current data.
    monitoring: Dict[str, ModelMonitoring]              # collection of monitoring objects.
    calculation_period_sec: float=15                    
    window_size: int

    def __init__(
        self, 
        datasets: Dict[str, LoadedDataset],
        window_size: int
    ):
        self.reference = {}
        self.monitoring = {}
        self.current = {}
        self.column_mapping = {}
        self.window_size = window_size

        for dataset_info in datasets.values():
            self.reference[dataset_info.name] = dataset_info.references
            self.monitoring[dataset_info.name] = ModelMonitoring(
                monitors=[EVIDENTLY_MONITORS_MAPPING[k]() for k in dataset_info.monitors], options=[]
            )
            self.column_mapping[dataset_info.name] = dataset_info.column_mapping
        
        self.metrics = {}
        self.next_run_time = {}
    

    def iterate(self, dataset_name: str, new_rows: pd.DataFrame):
        """Add data to current dataset for specified dataset."""
        window_size: self.window_size

        if dataset_name in self.current:
            current_data = self.current[dataset_name].append(new_rows, ignore_index=True)
        else:
            current_data = new_rows
        
        current_size = current_data.shape[0]

        if current_size > self.window_size:
            current_data.drop(index=list(range(0, current_size-self.window_size)), inplace=True) # Cut current size by window size. 
            current_data.reset_index(drop=True, inplace=True)                                    # Reset index.
        
        self.current[dataset_name] = current_data

        if current_size < window_size:
            logging.info(f"Not enough data for measurement: {current_size} of {window_size}" f" Waiting more data!")
            return