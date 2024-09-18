conda create -p myenv python=3.9 -y
# 1. Create app.py file
# 2. Add remote server for mlflow in degshub`

import dagshub
dagshub.init(repo_owner='debsandipagt', repo_name='mlflow', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)