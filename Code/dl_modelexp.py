import pandas as pd
import numpy as np
import yaml
import mlflow
import pickle as pkl
import os
from urllib.parse import urlparse
from typing import Literal

from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation, Input

from keras_tuner import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters


os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/Hegde-Darshan/kaggle_podcast_listentime.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = 'Hegde-Darshan'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'e246f6c3cb6893e7fdac9ad222326adeaf54e4b4'

def get_params(file: str ,key: str):
    return yaml.safe_load(open(file))[key]

def model_init(input_shape):

    model = Sequential()
    model.add(Input(shape=(input_shape[1],)))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'], initializer='he_normal')
    print(model.summary())
    return model

 
def dl_experiment(X, y, output_path, params):
        
        input_shape = X.shape
        model = model_init(input_shape)
    
        model.fit(
             X, y, 
                  epochs=100, 
                  batch_size=128, 
                  validation_split=0.2,
                  verbose=1, 
                  )

        
        mlflow.end_run()
        mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
        mlflow.set_experiment(experiment_name=params['experiment_name'])
        exp_id = mlflow.get_experiment_by_name(params['experiment_name']).experiment_id

        client = mlflow.tracking.MlflowClient()
        existing_run = client.search_runs(
            experiment_ids=exp_id, 
            run_view_type= ViewType.ACTIVE_ONLY, 
            filter_string= f"tags.mlflow.runName = 'dl_{len(model.layers)}_layers'",
            max_results=1
        )

        if not existing_run:
            mlflow.start_run(run_name=f"dl_{len(model.layers)}_layers")
            mlflow.start_run(nested=True)
        else:
            parent_run = existing_run[0]
            parent_run_id = parent_run.info.run_id
            mlflow.start_run(run_id= parent_run_id)
            mlflow.start_run(nested=True)
        
        # Save the model to MLflow
        #mlflow.keras.log_model(model, "model")
    
        # Log the metrics
        mlflow.log_metric("loss", model.history.history['loss'][-1])
        mlflow.log_metric("val_loss", model.history.history['val_loss'][-1])
        mlflow.log_metric("mae", model.history.history['mae'][-1])
        mlflow.log_metric("val_mae", model.history.history['val_mae'][-1])
        
        # Save the model locally
        model.save(os.path.join(output_path, f'model_{mlflow.active_run().info.run_id}.keras'))


if __name__ == "__main__":
         
    params = get_params('parameters.yaml', 'dl_experimentation')
    
    # Load your data here
    data = pd.read_csv(params['data'])
    X, y = data.drop(columns=['Listening_Time_minutes']).values, data[['Listening_Time_minutes']].values

    #X, y = X.transpose(), y.transpose()
    print(X.shape, y.shape)

    dl_experiment(X, y, params['models'], params)