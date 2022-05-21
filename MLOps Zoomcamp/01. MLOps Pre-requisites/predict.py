# IMPORTING MODULES:
print("Importing modules!")
import pickle

# LOADING THE MODEL:
input_file = "model_C=1.0.bin"

with open(input_file, "rb") as f_in:
    (dv, model) = pickle.load(f_in)  # Loading model.


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
y_pred = model.predict_proba(X)[0, 1]
print("input:", customer)
print("churn probability:", y_pred)