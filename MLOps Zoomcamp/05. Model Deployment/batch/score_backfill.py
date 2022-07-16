# IMPORTING MODULES:
import score 
from datetime import datetime  
from prefect import flow
from dateutil.relativedelta import relativedelta


@flow
def ride_duration_prediction_backfill():
    start_date = datetime(year=2021, month=3, day=1)
    end_date = datetime(year=2022, month=4, day=1)

    d = start_date

    while d <= end_date:
        score.ride_duration_prediction(
            taxi_type = "green", 
            run_id = "895c76c3ee6746cb96328b189042f646",
            run_date = d
        )

        d = d + relativedelta(months=1)


if __name__ == "__main__":
    ride_duration_prediction_backfill()
