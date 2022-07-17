# IMPORTING MODULES:
import os
import pickle
from matplotlib import collections 
import requests
from pymongo import MongoClient
from flask import Flask, request, jsonify



# INITIALIZING VARIABLES:
MODEL_FILE = os.getenv('MODEL_FILE', 'lin_reg.bin')
EVIDENTLY_SERVICE_ADDRESS = os.getenv('EVIDENTLY_SERVICE', 'http://127.0.0.1:5000')
MONGODB_ADDRESS = os.getenv('MONGODB_ADDRESS', 'mongodb://127.0.0.1:27017')



# LOADING THE MODEL:
with open(MODEL_FILE, "rb") as f_in:
    dv, model = pickle.load(f_in)



# INITIALIZING FLASK APPLICATION:
app = Flask('duration')                                 # Creating a flask. 
mongo_client = MongoClient(MONGODB_ADDRESS)             # Initializing mongo client. 
db = mongo_client.get_database('prediction_service')    # Creating database. 
collection = db.get_collection('data')                  # Creating collection inside database. 


@app.route('/predict', methods=['POST'])
def predict():
    record = request.get_json()
    record['PU_DO'] = '%s_%s' % (record['PULocationID'], record['DOLocationID'])
    X = dv.transform(record)
    y_pred = model.predict(X)

    result = {
        'duration': float(y_pred)
    }

    save_to_db(record, float(y_pred))
    send_to_evidently_service(record, float(y_pred))

    return jsonify(result)



# SAVING TO THE DATABASE:
def save_to_db(record, prediction):
    rec = record.copy()
    rec['prediction'] = prediction
    collection.insert_one(rec)



# SENDING TO EVIDENTLY_SERVICE:
def send_to_evidently_service(record, prediction):
    rec = record.copy()
    rec['prediction'] = prediction
    requests.post(f'{EVIDENTLY_SERVICE_ADDRESS}/iterate/taxi', json=[rec])



if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0, port=9696')