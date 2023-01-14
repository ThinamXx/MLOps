#!/usr/bin/env python
# coding: utf-8

# IMPORTING MODULES:
import os
import grpc
import tensorflow as tf

from keras_image_helper import create_preprocessor
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from flask import Flask
from flask import request
from flask import jsonify


# INITIALIZING TENSORFLOW-SERVING CHANNEL:
host = os.getenv("TF_SERVING_HOST", "localhost:8500")
channel = grpc.insecure_channel(host)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)


# CREATING PREPROCESSOR:
preprocessor = create_preprocessor("xception", target_size=(299, 299))


# CONVERTING NP ARRAY TO PROTOBUF:
def np_to_protobuf(data):
    return tf.make_tensor_proto(data, shape=data.shape)


# INITIALIZING THE PREDICTION:
def prepare_request(X):
    pb_request = predict_pb2.PredictRequest()

    pb_request.model_spec.name = "clothing-model"
    pb_request.model_spec.signature_name = "serving_default"
    pb_request.inputs["input_8"].CopyFrom(np_to_protobuf(X))

    return pb_request


# INITIALIZING THE RESPONSE:
classes = [
    "dress",
    "hat",
    "longsleeve",
    "outwear",
    "pants",
    "shirt",
    "shoes",
    "shorts",
    "skirt",
    "t-shirt",
]

def prepare_response(pb_response):
    preds = pb_response.outputs["dense_7"].float_val
    return dict(zip(classes, preds))


def predict(url):
    X = preprocessor.from_url(url)
    pb_request = prepare_request(X)
    pb_response = stub.Predict(pb_request, timeout=20.0)
    respose = prepare_response(pb_response)
    return respose


app = Flask('gateway')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    url = data["url"]
    result = predict(url)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
