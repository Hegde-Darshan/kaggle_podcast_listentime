import mlflow.entities
import pandas as pd
import numpy as np
import yaml
import mlflow
import pickle as pkl
import os
from urllib.parse import urlparse
from typing import Literal

from sklearn.metrics import root_mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn import svm
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split

os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/Hegde-Darshan/kaggle_podcast_listentime.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = 'Hegde-Darshan'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'e246f6c3cb6893e7fdac9ad222326adeaf54e4b4'

def calc_metrics(y_true, y_pred, metrics: Literal['all', 'accuracy', 'rmse']='all'):

    if metrics=='accuracy': 
        return {'accuracy': accuracy_score(y_true, y_pred)}
    elif metrics=='rmse'  : 
        return {'rmse': root_mean_squared_error(y_true, y_pred)}
    elif metrics=='mae':
        return {'mae': mean_absolute_error(y_true, y_pred)}
    else:
        return {'accuracy': accuracy_score(y_true, y_pred),
            'rmse': root_mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred)}
    

def model_init(params:dict, model_type:Literal['SGDRegressor', 'svm', 'adaboost']='SGDRegressor'):

    if model_type=='SGDRegressor':
        return SGDRegressor(
        loss= params['exp_models']['sgd_regressor']['loss'], 
        penalty=params['exp_models']['sgd_regressor']['penalty'],
        max_iter= params['exp_models']['sgd_regressor']['iterations']
    )
    elif model_type=='svm':
        return svm.LinearSVR( max_iter= params['exp_models']['svm']['iterations'])
    elif model_type=='adaboost':
        return AdaBoostRegressor(n_estimators= params['exp_models']['adaboost']['n_estimators'])
    else:
        return None


def get_params(file: str ,key: str):
    return yaml.safe_load(open(file))[key]


def ml_experiment(X, y, output_path, model_type:Literal['SGDRegressor', 'svm', 'adaboost'], params):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    mlflow.end_run()
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    mlflow.set_experiment(experiment_name="podcast_listentime")

    client = mlflow.tracking.MlflowClient()
    existing_run = client.search_runs(
        experiment_ids=0, 
        run_view_type= mlflow.entities.ViewType.ACTIVE_ONLY, 
        filter_string= f"tags.mlflow.runName = 'sklearn_{model_type}'",
        max_results=1
    )

    if not existing_run:
        mlflow.start_run(run_name='sklearn_'+model_type)
    else:
        parent_run = existing_run[0]
        parent_run_id = parent_run.info.run_id
        mlflow.start_run(run_id= parent_run_id)
        mlflow.start_run(nested=True)
    
    model = model_init(params, model_type)
    model.fit(X_train, y_train)

    mlflow.log_params(model.get_params())
    
    test_metrics = calc_metrics(y_test, model.predict(X_test), 'rmse')
    mlflow.log_metrics(test_metrics)

    test_metrics = calc_metrics(y_test, model.predict(X_test), 'mae')
    mlflow.log_metrics(test_metrics)

    signature = mlflow.models.infer_signature(X_train, y_train)

    mlflow.sklearn.log_model(model, f"{model_type}_model", signature=signature)

    mlflow.end_run()
    
    #save model locally
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pkl.dump(model, open(os.path.join(output_path, model_type+'_model.pkl'), 'wb'))
    return None


if __name__ == "__main__":

    params = get_params('parameters.yaml', 'experimentation')

    data = pd.read_csv(params['data'])
    X, y = data.drop(columns=['Listening_Time_minutes']), data[['Listening_Time_minutes']]

    for each_type in ['SGDRegressor', 'svm']:
        ml_experiment(X, y, params['models'], each_type, params)
