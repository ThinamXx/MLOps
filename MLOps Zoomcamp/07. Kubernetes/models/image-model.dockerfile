FROM tensorflow/serving:2.7.0

COPY clothing-model /models/clothing-models/1
ENV MOODEL_NAME="clothing-model"