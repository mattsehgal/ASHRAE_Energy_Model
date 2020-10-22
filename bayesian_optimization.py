# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import lightgbm as lgbm
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype
from sklearn.preprocessing import LabelEncoder
import datetime
import gc
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings('ignore')

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
    
###############
#bayesian optimization
def best_params_lgbm(X, y, opt_params, init_points=2, optimization_round=20, n_folds=3, random_seed=0, cv_estimators=1000):
    
    categorical_features = ["building_id", "site_id", "meter", "primary_use", "weekend"]
    train_data = lgbm.Dataset(data=X, label=y, categorical_feature = categorical_features, free_raw_data=False)
    
    def run_lgbm(num_leaves, feature_fraction, bagging_fraction, max_depth, lambda_l1, lambda_l2, min_split_gain, min_child_weight,learning_rate):
        params = {"boosting": "gbdt",'application':'regression','num_iterations':cv_estimators, 'early_stopping_round':int(cv_estimators/5), 'metric':'rmse'}
        params["num_leaves"] = int(round(num_leaves))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['lambda_l1'] = max(lambda_l1, 0)
        params['lambda_l2'] = max(lambda_l2, 0)
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight
        params['learning_rate'] = learning_rate
        cv_res = lgbm.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=False, verbose_eval=cv_estimators, metrics=['rmse'])
        #searching for max, thus -min
        return -min(cv_res['rmse-mean'])
    
    params_finder = BayesianOptimization(run_lgbm, opt_params, random_state=2021)
    #optimize params
    params_finder.maximize(init_points=init_points, n_iter=optimization_round)
    #best params
    return params_finder.max

###############

PATH = '/Users/quant/Downloads/ashrae-energy-prediction/'

print('creating dfs')
df_train = pd.read_csv(PATH + 'train.csv')
df_train = df_train [ df_train['building_id'] != 1099 ]
df_train = df_train.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')
df_building = pd.read_csv(PATH + 'building_metadata.csv')
df_weather = pd.read_csv(PATH + 'weather_train.csv')
df_weather = fill_missing_weather(df_weather)

print('reducing mem')
df_train = reduce_mem(df_train, use_float16=True)
df_weather = reduce_mem(df_weather, use_float16=True)
df_building = reduce_mem(df_building, use_float16=True)

print('merging')
df_train = df_train.merge(df_building, left_on='building_id', right_on='building_id', how='left')
df_train = df_train.merge(df_weather, how='left', left_on=['site_id','timestamp'], right_on=['site_id','timestamp'])
del df_weather
gc.collect()

print('feature engineering')
df_train = feature_engineering(df_train)
df_train.head(20)

print('set features and targets')
target = np.log1p(df_train["meter_reading"])
features = df_train.drop('meter_reading', axis = 1)
del df_train
gc.collect()

#configuration
print('set configuration')
#ranges for hyperparameters
params_range = {
                'num_leaves': (1000, 1280),
                'feature_fraction': (0.7, 0.9),
                'bagging_fraction': (0.8, 1),
                'max_depth': (10, 11),
                'lambda_l1': (2, 5),
                'lambda_l2': (2, 5),
                'min_split_gain': (0.001, 0.1),
                'min_child_weight': (5, 50),
                'learning_rate' : (.05,.07)
               }

#num folds for cv
folds = 3

#num iterations
cv_estimators = [1000, 1500, 2000]

#num models tested
optimization_round = 5 

#num steps of random exploration (diversifies exploration space) 
init_points = 2
#seed for random generation
random_seed = 2010

#find best parameters
print('find best params')
best_params= []
for cv_estimator in cv_estimators:
    opt_params = best_params_lgbm(features, target, params_range, init_points=init_points, optimization_round=optimization_round, n_folds=folds, random_seed=random_seed, cv_estimators=cv_estimator)
    opt_params['params']['iteration'] = cv_estimator
    opt_params['params']['fold'] = folds
    opt_params['params']['rmse'] = opt_params['target']
    best_params.append(opt_params['params'])

#best params to csv file
print('write params to file')
df_best_params = pd.DataFrame(best_params).reset_index()
df_best_params = df_best_params[['iteration','fold','num_leaves','learning_rate','bagging_fraction',
 'feature_fraction',
 'lambda_l1',
 'lambda_l2',
 'max_depth',
 'min_child_weight',
 'min_split_gain',
 'rmse']]

df_best_params.to_csv(r'C:\Users\quant\101termproj\bayesian_opt_results.csv')
print('script complete')
