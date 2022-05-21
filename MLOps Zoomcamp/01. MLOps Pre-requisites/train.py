# IMPORTING MODULES:
print("Importing modules!")
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# INITIALIZING PARAMETERS:
C = 1.0
n_splits = 5
output_file = f"model_C={C}.bin"  # Initializing output file.

# PROCESSING THE DATASET:
print("Processing data!")
df = pd.read_csv('data.csv')  # Reading the dataset.
df.columns = df.columns.str.lower().str.replace(' ', '_')  # Preparing columns.
categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)  # Index of categorical columns.
for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')
df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')  # Converting numerical column.
df.totalcharges = df.totalcharges.fillna(0)
df.churn = (df.churn == 'yes').astype(int)
df.head()  # Inspecting dataframe.

# PREPARING THE DATASET:
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)  # Splitting the dataset.
numerical = ['tenure', 'monthlycharges', 'totalcharges']  # Numerical columns.
categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]  # Categorical columns.


# FUNCTION TO TRAIN THE MODEL:
def train(df_train, y_train, C=1.0):  # Defining function.
    dicts = df_train[categorical + numerical].to_dict(orient="records")  # Creating dictionary.
    dv = DictVectorizer(sparse=False)  # Initialization.
    X_train = dv.fit_transform(dicts)  # Vectorization.
    model = LogisticRegression(C=C, max_iter=1000)  # Initializing logistic regression.
    model.fit(X_train, y_train)  # Training the model.
    return dv, model


# FUNCTION FOR PREDICTION:
def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient="records")  # Creating dictionary.
    X = dv.transform(dicts)  # Vectorization.
    y_pred = model.predict_proba(X)[:, 1]  # Generating predictions.
    return y_pred


# INITIALIZING KFOLD CROSS VALIDATION:
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=2022)  # Initializing KFold.
scores = []
for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]  # Training data.
    df_val = df_full_train.iloc[val_idx]  # Validation data.
    y_train = df_train.churn.values
    y_val = df_val.churn.values
    dv, model = train(df_train, y_train, C=C)  # Training the model.
    y_pred = predict(df_val, dv, model)  # Getting predictions.
    auc = roc_auc_score(y_val, y_pred)  # Getting roc auc.
    scores.append(auc)
print("C=%s %.3f +- %.3f" % (C, np.mean(scores), np.std(scores)))  # Inspection.

# INSPECTING SCORES:
scores

# TRAINING FINAL MODEL:
print("Training model!")
dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)  # Training the model.
y_pred = predict(df_test, dv, model)  # Model predictions.
auc = roc_auc_score(df_test.churn.values, y_pred)  # Inspecting auc roc.
auc
print(f"auc={auc}")

# SAVING THE MODEL:
f_out = open(output_file, "wb")  # Opening file.
pickle.dump((dv, model), f_out)
f_out.close()  # Closing file.
print("Model saved!")


# LOADING THE MODEL:
input_file = "model_C=1.0.bin"
with open(input_file, "rb") as f_in:
    (dv, model) = pickle.load(f_in)  # Loading model.
dv, model  # Inspection.

# INITIALIZING MODEL EVALUATION:
customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}
X = dv.transform([customer])

# MODEL PREDICTION:
model.predict_proba(X)[0, 1]
