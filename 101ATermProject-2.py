#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:20:18 2019

@authors: andrew, matt
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgbm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from lightgbm import LGBMRegressor
import datetime
import os
import gc

PATH = "../input/ashrae-energy-prediction/"

df_train = pd.read_csv(PATH + 'train.csv')
df_building = pd.read_csv(PATH + 'building_metadata.csv')
df_weather = pd.read_csv(PATH + 'weather_train.csv')
df_train = df_train[df_train['building_id'] != 1099]
df_train = df_train.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')
df_train = df_train.merge(df_building, on='building_id', how='left')
df_train = df_train.merge(df_weather, on=['site_id', 'timestamp'], how='left')
df_train.timestamp = pd.to_datetime(df_train.timestamp, format="%Y-%m-%d %H:%M:%S")

#TODO: fill in missings

from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype

def reduce_mem(df, use_float16=False):
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.        
    """
    
    start_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))
    
    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            continue
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))
    
    return df

df_train = reduce_mem(df_train,use_float16=True)
del df_building
del df_weather
gc.collect()

le = LabelEncoder()
df_train.primary_use = le.fit_transform(df_train.primary_use)
test_feature = df_train[df_train['timestamp'] >= pd.to_datetime('2016-10-01')]
train_feature = df_train[df_train['timestamp'] < pd.to_datetime('2016-10-01')]
test_target = test_feature['meter_reading']
train_target = np.log1p(train_feature['meter_reading'])
drop_features = ['timestamp', 'meter_reading', 'year_built', 'floor_count', 'sea_level_pressure', 'wind_direction', 'wind_speed']
test_feature = test_feature.drop(columns = drop_features)
train_feature = train_feature.drop(columns = drop_features)
del df_train
gc.collect()

categorical_features = ["building_id", "site_id", "meter", "primary_use"]
model = LGBMRegressor()
model.fit(train_feature, train_target, categorical_feature=categorical_features)
print(np.sqrt(mean_squared_log_error(test_target, np.clip(np.expm1(model.predict(test_feature)), 0, None))))