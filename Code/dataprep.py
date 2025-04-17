import pandas as pd
import numpy as np
import yaml
import os
import pickle as pkl

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def get_params(file: str ,key: str):
    return yaml.safe_load(open(file))[key]

def preprocess(input_path, output_path, output_models):
    #read data
    data = pd.read_csv(input_path)

    #discard unimportant features:
    data.drop(columns=['id', 'Episode_Title'], inplace=True)

    #drop rows with outliers in popularity percentage
    data.drop(index=data[(data['Host_Popularity_percentage']>100) | (data['Guest_Popularity_percentage']>100)].index, inplace=True)
    data.replace(['','NaN', pd.NA], np.nan, inplace=True)
    #impute null in Episode_Length and Guest_Popularity
    #Onehotencode Podcast_Name, Genre, Publication_Time, Publication_Day, episode_sentiment
    #standard scale episode_length, host_popularity, guest_popularity
    #all the above in a single column transformer
    """
    ['id', 'Podcast_Name', 'Episode_Title', 'Episode_Length_minutes',
       'Genre', 'Host_Popularity_percentage', 'Publication_Day',
       'Publication_Time', 'Guest_Popularity_percentage', 'Number_of_Ads',
       'Episode_Sentiment', 'Listening_Time_minutes']
    """
    #imputing before column transformations
    imputer = ColumnTransformer(
        [
            ("imputer_mean", SimpleImputer(strategy='mean', missing_values=np.nan), ['Episode_Length_minutes']),
            ("imputer_median", SimpleImputer(strategy="median",  missing_values=np.nan), ['Guest_Popularity_percentage', 'Host_Popularity_percentage','Number_of_Ads']),
        ], remainder="passthrough", verbose=True, verbose_feature_names_out=False)
    imputer.set_output(transform="pandas")

    transformers = ColumnTransformer(
        [
            
            ("Scaling_Standard", StandardScaler(), ['Episode_Length_minutes','Host_Popularity_percentage','Guest_Popularity_percentage']),
            ("Categ_Encoder_OneHot", OneHotEncoder(drop='first', handle_unknown='infrequent_if_exist', sparse_output=False), ['Podcast_Name', 'Genre','Publication_Day','Publication_Time','Episode_Sentiment'])
        ], remainder='passthrough', verbose=True, verbose_feature_names_out=False)
    transformers.set_output(transform="pandas")

    transformation_pipe = Pipeline(
        [
            ('imputation', imputer),
            ('transformation', transformers)
        ])
    
    transformed_data = transformation_pipe.fit_transform(data)

    print(transformers.get_feature_names_out())
    #transformed_data = pd.DataFrame(transformed_cols, columns=transformers.get_feature_names_out())
    #print(transformed_data[transformed_data.isnull().any(axis=0)].shape)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    transformed_data.to_csv(os.path.join(output_path, 'transformed.csv'), index=False)

    os.makedirs(os.path.dirname(output_models), exist_ok=True)
    pkl.dump(transformers, open(os.path.join(output_models, 'preprocessor.pkl'), 'wb'))


if __name__ == "__main__":
    params = get_params("parameters.yaml", "preprocess")

    preprocess(params['input'], params['output'], params['models'])
