# IMPORTING MODULES:
import requests
import predict


# DEFINING THE EXAMPLE:
ride = {
    "PULocationID": 10,
    "DOLocationID": 50,
    "trip_distance": 40
}


# MAKING THE PREDICTION:
url = "http://localhost:9696/predict"
response = requests.post(url, json=ride)
print(response.json())