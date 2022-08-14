# IMPORTING MODULES:
import warnings
import datetime
import numpy as np
import mlflow.sklearn
import pandas_datareader.data as web

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics import precision_score, recall_score


# GETTING TRAINING DATA:
def acquire_training_data():
    """
    Acquire training data from Yahoo API using pandas data reader. 
    """
    start = datetime.datetime(2019, 7, 1)
    end = datetime.datetime(2019, 9, 30)
    df = web.DataReader("BTC-USD", "yahoo", start, end)
    return df


# DEPENDENCY FUNCTION:
def digitize(n):
    if n > 0: 
        return 1
    return 0


# DEPENDENCY FUNCTION:
def rolling_window(arr, window):
    """
    Args:
        arr (np.array): Array to be rolled over.
        window (int): Window size.
    Returns:
        arr (np.array): Array with all the ordered sequences of values of 'arr' of size 'window'.
    """
    shape = arr.shape[:-1] + (arr.shape[-1] - window + 1, window)
    strides = arr.strides + (arr.strides[-1], )
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)



# PREPARING THE TRAINING DATASET: 
def prepare_training_data(data):
    """
    Prepare training data for training and validation.
    Args:
        data (pd.DataFrame): A dataframe containing training data.
    Returns:
        data (np.DataFrame): A numpy dataframe containing training data.
    """
    data["Delta"] = data["Close"] - data["Open"]
    data["to_predict"] = data["Delta"].apply(lambda x: digitize(x))
    return data



if __name__ == "__main__":
    warnings.filterwarnings("ignore") # Ignore warnings.

    with mlflow.start_run():
        training_data = acquire_training_data() # Get training data.
        prepared_training_data = prepare_training_data(training_data) # Get prepared training data.
        data_mat = prepared_training_data.to_numpy() # Converting into numpy array format. 

        WINDOW_SIZE = 14

        X = rolling_window(data_mat[:, 7], WINDOW_SIZE)[:-1, :]
        Y = prepared_training_data["to_predict"].to_numpy()[WINDOW_SIZE:] # Converting into numpy array format.
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=2022, stratify=Y)

        clf = RandomForestClassifier(
            bootstrap=True, criterion="gini",
            min_samples_leaf=2,
            min_weight_fraction_leaf=0.0,
            n_estimators=50,
            random_state=2022,
            verbose=0
        ) # Initializing the classifier.
        clf.fit(X_train, y_train) # Training the classifier model.
        predicted = clf.predict(X_test)

        mlflow.sklearn.log_model(clf, "model_random_forest")

        print(classification_report(y_test, predicted))

        mlflow.log_metric("precision_label_0", precision_score(y_test, predicted, pos_label=0))
        mlflow.log_metric("precision_label_1", precision_score(y_test, predicted, pos_label=1))
        mlflow.log_metric("recall_label_0", recall_score(y_test, predicted, pos_label=0))
        mlflow.log_metric("recall_label_1", recall_score(y_test, predicted, pos_label=1))
        mlflow.log_metric("f1score_label_0", f1_score(y_test, predicted, pos_label=0))
        mlflow.log_metric("f1score_label_1", f1_score(y_test, predicted, pos_label=1))