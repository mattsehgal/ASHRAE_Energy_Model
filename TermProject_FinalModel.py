#!/usr/bin/env python3
import pandas as pd
import numpy as np
import lightgbm as lgbm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import datetime
import gc

#FUNCTIONS
###############
#reduce memory usage function
def reduce_mem(df, use_float16=False):
    
    start_mem = df.memory_usage().sum() / 1024 ** 2
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

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df
###############

###############
#fill missing weather data function
def fill_missing_weather(df_weather):
    #fill missing dates
    format = "%Y-%m-%d %H:%M:%S"
    first_date = datetime.datetime.strptime(df_weather['timestamp'].min(), format)
    last_date = datetime.datetime.strptime(df_weather['timestamp'].max(), format)
    hrs_total = int(((last_date - first_date).total_seconds() + 3600) / 3600)
    hrs_list = [(last_date - datetime.timedelta(hours = h)).strftime(format) for h in range(hrs_total)]

    missing_hours = []
	#16 sites w hourly data
    for id in range(16):
        hours = np.array(df_weather[df_weather['site_id'] == id]['timestamp'])
        add_rows = pd.DataFrame(np.setdiff1d(hrs_list, hours), columns=['timestamp'])
        add_rows['site_id'] = id
        df_weather = pd.concat([df_weather, add_rows],sort=True)
        df_weather = df_weather.reset_index(drop=True)           

    #add new date/time features
    df_weather["datetime"] = pd.to_datetime(df_weather["timestamp"])
    df_weather["day"] = df_weather["datetime"].dt.day
    df_weather["week"] = df_weather["datetime"].dt.week
    df_weather["month"] = df_weather["datetime"].dt.month
    
    #index reset, fast update
    df_weather = df_weather.set_index(['site_id','day','month'])
	#fill empties
	#fill air_temperature
    fill_air_temp = pd.DataFrame(df_weather.groupby(['site_id','day','month'])['air_temperature'].mean(), columns=["air_temperature"])
    df_weather.update(fill_air_temp,overwrite=False)
	#fill cloud_coverage
    fill_cloud_cover = df_weather.groupby(['site_id','day','month'])['cloud_coverage'].mean()
    fill_cloud_cover = pd.DataFrame(fill_cloud_cover.fillna(method='ffill'), columns=["cloud_coverage"])
    df_weather.update(fill_cloud_cover,overwrite=False)
	#fill dew_temperature
    fill_dew_temp = pd.DataFrame(df_weather.groupby(['site_id','day','month'])['dew_temperature'].mean(), columns=["dew_temperature"])
    df_weather.update(fill_dew_temp, overwrite=False)
	#fill sea_level_pressure
    fill_sea_level = df_weather.groupby(['site_id','day','month'])['sea_level_pressure'].mean()
    fill_sea_level = pd.DataFrame(fill_sea_level.fillna(method='ffill'), columns=['sea_level_pressure'])
    df_weather.update(fill_sea_level, overwrite=False)
	#fill wind_direction
    fill_wind_dir =  pd.DataFrame(df_weather.groupby(['site_id','day','month'])['wind_direction'].mean(), columns=['wind_direction'])
    df_weather.update(fill_wind_dir, overwrite=False)
	#fill wind_speed
    fill_wind_speed =  pd.DataFrame(df_weather.groupby(['site_id','day','month'])['wind_speed'].mean(), columns=['wind_speed'])
    df_weather.update(fill_wind_speed, overwrite=False)
	#fill precip_depth_1_hr
    fill_precip_depth = df_weather.groupby(['site_id','day','month'])['precip_depth_1_hr'].mean()
    fill_precip_depth = pd.DataFrame(fill_precip_depth.fillna(method='ffill'), columns=['precip_depth_1_hr'])
    df_weather.update(fill_precip_depth, overwrite=False)

    df_weather = df_weather.reset_index()
    df_weather = df_weather.drop(['datetime','day','week','month'], axis=1) 
    return df_weather
###############

###############
#feature engineering
def feature_engineering(df):
    #sort by timestamp
    df.sort_values("timestamp")
    df.reset_index(drop=True)
    
    #add features
    df["timestamp"] = pd.to_datetime(df["timestamp"],format="%Y-%m-%d %H:%M:%S")
    df["hour"] = df["timestamp"].dt.hour
    df["weekend"] = df["timestamp"].dt.weekday
    df['square_feet'] =  np.log1p(df['square_feet'])
    
    #remove unused cols
    drop = ["timestamp","sea_level_pressure", "wind_direction", "wind_speed","year_built","floor_count"]
    df = df.drop(drop, axis=1)
    gc.collect()
    
    #LE on categorical
    le = LabelEncoder()
    df["primary_use"] = le.fit_transform(df["primary_use"])
    
    return df
###############
    
print('Beginning run...\n')

pd.set_option('display.max_columns', 20)

PATH = '/Users/quant/Downloads/ashrae-energy-prediction/'
print('>> Creating DataFrames...')
df_train = pd.read_csv(PATH + 'train.csv')
df_train = df_train [ df_train['building_id'] != 1099 ]
df_train = df_train.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')
df_building = pd.read_csv(PATH + 'building_metadata.csv')
df_weather = pd.read_csv(PATH + 'weather_train.csv')

#fill weather
print('>> Filling missing weather values...')
df_weather = fill_missing_weather(df_weather)

#reduce mem
print('>> Reducing DataFrame memory usage...\n')
df_train = reduce_mem(df_train, use_float16=True)
df_weather = reduce_mem(df_weather, use_float16=True)
df_building = reduce_mem(df_building, use_float16=True)

#merge data
print('\n>> Merging training data...')
df_train = df_train.merge(df_building, left_on='building_id', right_on='building_id', how='left')
df_train = df_train.merge(df_weather, how='left', left_on=['site_id','timestamp'], right_on=['site_id','timestamp'])
del df_weather
gc.collect()

#feature engineering
print('>> Running feature engineering...')
df_train = feature_engineering(df_train)
df_train.head(20)

#features & target vars
print('>> Setting features and target variables...')
target = np.log1p(df_train["meter_reading"])
features = df_train.drop('meter_reading', axis = 1)
del df_train
gc.collect()

#kfold lgbm model
print('>> Training models with KFold...\n')
categorical_features = ["building_id", "site_id", "meter", "primary_use", "weekend"]
params = {
    "objective": "regression",
    "boosting": "gbdt",
    "num_leaves": 1280,
    "learning_rate": 0.05,
    "feature_fraction": 0.85,
    "reg_lambda": 2,
    "metric": "rmse",
}

kfold = KFold(n_splits=3)
models = []

for train_index, test_index in kfold.split(features):
    features_train = features.loc[train_index]
    target_train = target.loc[train_index]
    
    features_test = features.loc[test_index]
    target_test = target.loc[test_index]
    
    d_training = lgbm.Dataset(features_train, label=target_train,categorical_feature=categorical_features, free_raw_data=False)
    d_test = lgbm.Dataset(features_test, label=target_test,categorical_feature=categorical_features, free_raw_data=False)
    
    model = lgbm.train(params, train_set=d_training, num_boost_round=1000, valid_sets=[d_training,d_test], verbose_eval=25, early_stopping_rounds=50)
    models.append(model)
    
    del features_train, target_train, features_test, target_test, d_training, d_test
    gc.collect()
	
del features, target
gc.collect()

#feature importance
print('\n>> Plotting feature importance...')
for model in models:
    lgbm.plot_importance(model)
    plt.show()

print('>> Beginning testing...')

#load test data
print('>> Loading test data...\n')
df_test = pd.read_csv(PATH + 'test.csv')
row_id = df_test["row_id"]
df_test.drop("row_id", axis=1, inplace=True)
df_test = reduce_mem(df_test)

#merge building data
print('\n>> Merging building data in test set...')
df_test = df_test.merge(df_building, left_on='building_id', right_on='building_id', how='left')
del df_building
gc.collect()

#fill weather
print('>> Filling missing weather values...\n')
df_weather = pd.read_csv(PATH + 'weather_test.csv')
df_weather = fill_missing_weather(df_weather)
df_weather = reduce_mem(df_weather)

#merge weather data
print('\n>> Merging weather data...')
df_test = df_test.merge(df_weather,how='left',on=['timestamp','site_id'])
del df_weather
gc.collect()

#feature engineering
print('>> Running feature engineering...')
df_test = feature_engineering(df_test)
df_test.head(20)

#predict
print('>> Beginning prediction...')
x = 0
results = []
for model in models:
    x = x+1
    print(' > Creating prediction model #',x,'...')
    if results == []:
        results = np.expm1(model.predict(df_test, num_iteration=model.best_iteration)) / len(models)
    else:
        results += np.expm1(model.predict(df_test, num_iteration=model.best_iteration)) / len(models)
    del model
    gc.collect()
    print(' > Model #',x,' has been created...')
	
del df_test, models
gc.collect()

#results for submission
print('\n>> Creating results DataFrame...')
df_results = pd.DataFrame({"row_id": row_id, "meter_reading": np.clip(results, 0, a_max=None)})

del row_id, results
gc.collect()
print('>> Writing results to submission file...')
df_results.to_csv(r'C:\Users\quant\101termproj\submission.csv', index=False)

print('>> Validating number of rows...')
if len(df_results.index) == 41697600:
    print(' > Submission file is VALID, terminating run...')
else:
    print(' > Submission file is INVALID, terminating run...')
print('\nRun terminated.')









