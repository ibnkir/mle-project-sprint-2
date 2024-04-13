# scripts/fit.py

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from category_encoders import CatBoostEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
import yaml
import os
import joblib


def fit_model():
    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd)

    data = pd.read_csv('data/initial_data.csv')

    cat_features = data.select_dtypes(include=['bool', 'object'])
    is_binary_cat_features = cat_features.nunique() == 2
    binary_cat_features = cat_features[is_binary_cat_features[is_binary_cat_features].index]
    other_cat_features = cat_features[is_binary_cat_features[~is_binary_cat_features].index]
    num_features = data.select_dtypes(['float']) #.drop(columns=params['target_col'])
   
    preprocessor = ColumnTransformer(
        [
            ('binary_cat', OneHotEncoder(drop=params['one_hot_drop']), binary_cat_features.columns.tolist()),
            ('other_cat', CatBoostEncoder(), other_cat_features.columns.tolist()),
            ('num', StandardScaler(), num_features.columns.tolist())
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
  
    #model = LinearRegression(n_jobs=params['n_jobs'])
    #model = Lasso(alpha=params['alpha'], max_iter=params['max_iter'])
    #model = Ridge(alpha=params['alpha'])
    #model = GradientBoostingRegressor(random_state=params['random_state'])
    model = CatBoostRegressor(loss_function='MAPE', random_state=params['random_state'])
    
    pipeline = Pipeline(
        [
            ('preprocessor', preprocessor),
            ('model', model)
        ]
    )
    pipeline.fit(data, data[params['target_col']])
 
    os.makedirs('models', exist_ok=True)
    with open('models/fitted_model.pkl', 'wb') as fd:
        joblib.dump(pipeline, fd) 


if __name__ == '__main__':
    fit_model()
