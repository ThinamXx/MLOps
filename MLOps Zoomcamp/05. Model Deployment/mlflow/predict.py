# IMPORTING MODULES:
import mlflow
import pickle
import pandas as pd
from pyexpat import model
from flask import Flask, request, jsonify
from mlflow.tracking import MlflowClient



# INTERACTING WITH MODEL REGISTRY:
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
run_id = "e46fafe797f94d07a522ad82dc1b5af8"

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


path = client.download_artifacts(run_id=run_id, path="dict_vectorizer.bin")
with open(path, 'rb') as f_out:
    dv = pickle.load(f_out)

logged_model = f"runs:/{run_id}/model"
model = mlflow.pyfunc.load_model(logged_model)


# FUNCTION TO PREPARE THE FEATURES:
def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])  
    features['trip_distance'] = ride['trip_distance']
    return features



# DEFINE PREDICTION FUNCTION:
def predict(features):
    X = dv.transform(features)
    preds = model.predict(X)
    return float(preds[0])



# INITIALIZATION:
app = Flask('duration-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()
    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'duration': pred 
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(port=9696, debug=True, host='0.0.0.0')