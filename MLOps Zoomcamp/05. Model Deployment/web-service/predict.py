# IMPORTING MODULES:
import pickle
from pyexpat import model
from flask import Flask, request, jsonify

# LOADING THE MODEL:
with open("./lin_reg.bin", "rb") as f_in:
    (dv, model) = pickle.load(f_in)



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