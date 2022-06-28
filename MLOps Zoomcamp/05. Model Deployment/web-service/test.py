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
features = predict.prepare_features(ride)
pred = predict.predict(features)
print(pred)