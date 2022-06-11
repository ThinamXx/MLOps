# **Experiment Tracking & Model Management**

### **Experiment Tracking**
- Experiment tracking is the process of keeping track of all the relevant information from an **ML experiment**, which includes source code, environment, data, model, hyperparameters, metrics, etc.


### **MLflow**
- An open source platform for the machine learning lifecycle. It's a python package that can be installed with pip and it contains four main modules:
	- Tracking.
	- Models.
	- Model Registry.
	- Projects.

### **Model Management**
- Tracks model versions and model artifacts along with preprocessor artifacts. 

### **Model Registry**
- List the production ready models. 
- Deployment of latest versions of the model.

### **MLflowClient Class**
- A client of an MLflow Tracking Server creates and manages experiments andruns.
- A client of an MLflow Registry Server creates and manages registered models and model versions. 

### **MLflow Configuration**
- MLflow configuration is a set of key-value pairs that are used to configure MLflow.
- MLflow configuration is stored in a file called **mlflow-config.yaml**.