# Pandas will be the backbone of our data manipulation.
import pandas as pd 
from pandas import json_normalize
# Seaborn is a data visualization library.
import seaborn as sns
# Matplotlib is a data visualization library.
# Seaborn is actually built on top of Matplotlib.
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# Numpy will help us handle some work with arrays.
import numpy as np
# Datetime will allow Python to recognize dates as dates, not strings.
from datetime import datetime
import strava_api

activities = json_normalize(strava_api.my_dataset)
print (activities.columns) #See a list of all columns in the table
print (activities.shape) #See the dimensions of the table.
print(activities.dtypes) #See the data types of each column.

print (activities['start_date_local'])

# Create new dataframe with only columns I care about
cols = ['name',
        'average_speed', 'suffer_score', 'upload_id', 'type', 'distance', 'moving_time', 'max_speed', 'total_elevation_gain',
        'start_date_local', 'average_heartrate', 'max_heartrate', 'workout_type', 'elapsed_time', 'average_cadence'
        ]
activities = activities[cols]  # Break date into start time and date
activities['start_date_local'] = pd.to_datetime(activities['start_date_local'])

activities['weekday'] = activities['start_date_local'].map(lambda x: x.weekday)
activities['start_time'] = activities['start_date_local'].dt.time

print(activities.dtypes) #See the data types of each column.

# activities['start_time'] = activities['start_date_local'].dt.time.apply(
    # lambda x: x.strftime('%H:%M:%S'))
# activities['start_date_local'] = activities['start_date_local'].dt.date
activities.head(5)

print(activities.head(5))

#runs = activities.loc[activities['type'] == 'Run' & activities['start_date_local'].dt.year == 2023]
runs = activities.loc[(activities['type'] == 'Run') & (activities['start_date_local'].dt.year == 2023)]
runs['distance'] = runs['distance'] * 0.000621371
runs['pace'] = (runs['moving_time'] / 60) / (runs['distance'])
runs['cadence'] = (runs['average_cadence'] * 2)
# runs["start_time"] = pd.to_datetime(runs["start_time"])
runs['start_time_unix'] = runs['start_date_local'].apply(
    lambda x: x.timestamp())
runs['start_time_unix'] = runs['start_date_local'].apply(
    lambda x: x.timestamp())
runs['start_time_str'] = runs['start_time'].apply(
    lambda x: x.strftime('%H:%M:%S'))
runs['start_time_hr_str'] = runs['start_time_str'].str[:2]
runs['start_time_hr_int'] = runs['start_time_hr_str'].apply(lambda x: int(x))
runs['moving_time_formatted'] = runs['moving_time'].apply(
    lambda x: f"{int(x/60):02d}:{int(x%60):02d}")
