# **Model Monitoring**

### **Machine Learning Model Monitoring**
1. Service Health
2. Model Performance
3. Data Quality & Integrity
4. Data Drift & Concept Drift
5. Performance by Segment
6. Model Bias & Fairness
7. Outliers
8. Explainability


### **Monitoring Pipeline**
`ML Service + Usage Simulation + Logging + Online Monitoring + Batch Monitoring + Container`


### **Monitoring Pipeline Instructions**

- Create a virtual environment and install the `requirements.txt` file.  
```python3
pipenv --python=3.10.4
pipenv shell
pipenv install -r requirements.txt
```  
  
- Run the commands:  
```python3
python3 prepare.py 
docker-compose up
```