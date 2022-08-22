# **Data Science Workbench**  

The basic elements of this folder are outlined below:  
- `Makefile`: This allows control of your workbench. By issuing commands, you can ask your workbench to set up a new environment notebook to start MLflow in different formats.  
- `README.md`: This file contains a description of your project and how to run it.
- `docker`: A folder that encloses the Docker images of the different subsystems that our environment consists of. 
- `docker-compose.yml`: A file that contains the orchestration of different services in our workbench environment: Jupyter Notebook, MLflow, and PostgreSQL to back MLflow.
- `src`: A folder that encloses the source code of the project, to be updated in further phases of the project.
- `tox.ini`: A templated file that controls the execution of the unit tests. 