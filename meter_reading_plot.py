#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 17:48:43 2019

@author: andrew
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_train = pd.read_csv('/Users/andrew/Desktop/gitdemoapp/101TermProject/ashrae-energy-prediction/train.csv')
fig, axes = plt.subplots(1,2, figsize = (12, 5), dpi = 100)
df_train.query('building_id <= 104 & meter == 0 & timestamp <= "2016-05-20"')['meter_reading'].plot.hist(ax=axes[0], title = 'Site 0 electrc meter reading up to 2016-05-20')
df_train.query('building_id <= 104 & meter == 0 & timestamp > "2016-05-20"')['meter_reading'].plot.hist(ax=axes[1], title = 'Site 0 electrc meter reading after 2016-05-20')
plt.show()

df_train.timestamp = pd.to_datetime(df_train.timestamp, format="%Y-%m-%d %H:%M:%S")
fig, axes = plt.subplots(3, 1, figsize=(12, 18), dpi=100)
df_train[['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes[0]).set_ylabel('Meter reading', fontsize=12)
axes[0].set_title('Mean meter reading by day', fontsize=12)
df_train[df_train['building_id']==1099][['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes[1]).set_ylabel('Meter reading', fontsize=12)
axes[1].set_title('Mean meter reading by day for building 1099', fontsize=12)
df_train = df_train[df_train['building_id'] != 1099]
df_train[['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes[2]).set_ylabel('Meter reading', fontsize=12)
axes[2].set_title('Mean meter reading by day excluding building 1099', fontsize=12)
plt.show()

df_weather = pd.read_csv('/Users/andrew/Desktop/gitdemoapp/101TermProject/ashrae-energy-prediction/weather_test.csv')
df_weather = df_weather[df_weather['site_id']==0][['timestamp', 'air_temperature']]
df_weather.timestamp = pd.to_datetime(df_weather.timestamp, format="%Y-%m-%d %H:%M:%S")
fig, axes = plt.subplots(1, 1, figsize=(12, 5), dpi=100)
df_weather[['timestamp', 'air_temperature']].set_index('timestamp').resample('H').mean()['air_temperature'].plot(ax=axes, label = 'By hour', alpha=0.8).set_ylabel('Air temperature', fontsize=12)
df_weather[['timestamp', 'air_temperature']].set_index('timestamp').resample('D').mean()['air_temperature'].plot(ax=axes, label = 'By day', alpha=0.8).set_ylabel('Air temperature', fontsize=12)
axes.set_title('Mean air temperature by hour and day', fontsize=12)
axes.legend()
plt.show()

df_test = pd.read_csv('/Users/andrew/Desktop/gitdemoapp/101TermProject/ashrae-energy-prediction/test.csv')
df_result = pd.read_csv('/Users/andrew/Desktop/gitdemoapp/101TermProject/submission.csv')
df_result = df_result.merge(df_test, on='row_id', how='left')
df_result = df_result.query('building_id <= 104')
df_result.timestamp = pd.to_datetime(df_result.timestamp, format="%Y-%m-%d %H:%M:%S")
fig, axes = plt.subplots(1, 1, figsize=(12, 5), dpi=100)
df_result[['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes, label = 'By hour', alpha=0.8).set_ylabel('Meter reading', fontsize=12)
df_result[['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes, label = 'By day', alpha=0.8).set_ylabel('Meter reading', fontsize=12)
axes.set_title('Mean meter reading by hour and day', fontsize=12)
axes.legend()
plt.show()