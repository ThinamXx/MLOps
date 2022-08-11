docker build -t mlflow_eng1 .

docker run -p 8888:8888 -p 5000:5000 -v "$(pwd)" mlflow_eng1