# IMPORTING MODULES:
import pickle
from pyexpat import model


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
    return preds