#!/usr/bin/env python3

import pandas as pd
import numpy as np

from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype

from sklearn.model_selection import KFold

pd.set_option('display.max_columns', 10)

PATH = '/Users/jacobhan/Documents/GitHub/101TermProject' # this is the location of your local git repo

df_train = pd.read_csv(PATH + '/ashrae-energy-prediction/train.csv')
df_weather_train = pd.read_csv(PATH + '/ashrae-energy-prediction/weather_train.csv')
df_weather_test = pd.read_csv(PATH + '/ashrae-energy-prediction/weather_test.csv')
df_building_metadata = pd.read_csv(PATH + '/ashrae-energy-prediction/building_metadata.csv')

def reduce_mem(df, use_float16 = False):

	initial_mem = df.memory_usage().sum() / 1024 ** 2
	print("initial df mem usage: {:.2f}mb".format(initial_mem))
  
	for col in df.columns:
		if is_datetime(df[col]) or is_categorical_dtype(df[col]):
			continue
		col_dtype = df[col].dtype
					   
		if col_dtype != object:
			col_min = df[col].min()
			col_max = df[col].max()
		  
			if str(col_dtype) [:3] == "int":
				if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
					df[col] = df[col].astype(np.int8)
				elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
					df[col] = df[col].astype(np.int16)
				elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
					df[col] = df[col].astype(np.int32)
				elif col_min > np.iinfo(np.int64) and col_max < np.iinfo(np.int64).max:
					df[col] = df[col].astype(int64)
			else:
				if use_float16 and col_min > np.iinfo(np.float16) and col_max < np.iinfo(np.col16):
					df[col] = df[col].astype(np.float16)
				elif col_min > np.finfo(np.float32).min and col_max < np.finfo(np.float32).max:
					df[col] = df[col].astype(np.float32)
				else:
					df[col] = df[col].astype(np.float64)
		else:
			df[col] = df[col].astype("category")

	final_mem = df.memory_usage().sum() / 1024 ** 2
	print("reduced df mem usage: {:.2f}mb, reduced by {:.1f} percent".format(final_mem, (100 * (initial_mem - final_mem) / initial_mem)))	

#

# data pre-processing - reduce memory usage

reduce_mem(df_train)
reduce_mem(df_building_metadata)
reduce_mem(df_weather_train)
reduce_mem(df_weather_test)

# training the regression models

# merge train and building metadata table
df_tbm = pd.merge(df_building_metadata, df_train, on='building_id')

print(df_tbm)

kf = KFold(n_splits=10, shuffle=True)
print(kf)

# generating output

#