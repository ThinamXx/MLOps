# IMPORTING MODULES:
import requests

host = 'churn-serving-env.eba-i6ay3kbz.us-west-2.elasticbeanstalk.com'
url = f'http://{host}/predict'

customer_id = 'xyz-123'
customer = {
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "no",
    "phoneservice": "no",
    "multiplelines": "no_phone_service",
    "internetservice": "dsl",
    "onlinesecurity": "no",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "tenure": 1,
    "monthlycharges": 29.85,
    "totalcharges": (1 * 29.85)
}


response = requests.post(url, json=customer).json()
print(response)

if response['churn']:
    print('sending promo email to %s' % customer_id)
else:
    print('not sending promo email to %s' % customer_id)
