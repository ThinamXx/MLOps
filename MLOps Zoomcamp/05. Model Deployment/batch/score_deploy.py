# INITIALIZING DEPLOYMENT:
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner



# DEFINING DEPLOYMENT SPEC:
DeploymentSpec(
    flow_location="score.py",
    name="ride_duration_prediction",
    parameters={
        "taxi_type": "green", 
        "run_id": "895c76c3ee6746cb96328b189042f646",
    },
    schedule=CronSchedule(cron="0 3 2 * *"),
    flow_runner=SubprocessFlowRunner(),
    flow_storage="76bf03b6-80cb-4c32-8730-da554657957c",
    tags=["ml"]
)