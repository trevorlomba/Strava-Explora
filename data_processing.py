import pandas as pd
from pandas import json_normalize
import strava_api
import strava_cleaning
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import pytz
import numpy as np
import uuid






def get_activities():
    activities = json_normalize(strava_api.my_dataset)

    # Create new dataframe with only columns I care about
    cols = ['name',
            'average_speed', 'suffer_score', 'upload_id', 'type', 'distance', 'moving_time', 'max_speed', 'total_elevation_gain',
            'start_date_local', 'average_heartrate', 'max_heartrate', 'workout_type', 'elapsed_time', 'average_cadence'
            ]
    activities = activities[cols]  # Break date into start time and date
    activities['start_date_local'] = pd.to_datetime(activities['start_date_local'])
    activities['weekday'] = activities['start_date_local'].map(lambda x: x.weekday)
    activities['start_time'] = activities['start_date_local'].dt.time
    runs = activities.loc[(activities['type'] == 'Run') & (activities['start_date_local'].dt.year == 2023)]
    runs['distance'] = runs['distance'] * 0.000621371
    runs['pace'] = (runs['moving_time'] / 60) / (runs['distance'])
    runs['cadence'] = (runs['average_cadence'] * 2)
    # runs["start_time"] = pd.to_datetime(runs["start_time"])
    runs['start_time_unix'] = runs['start_date_local'].apply(lambda x: x.timestamp())
    runs['start_time_unix'] = runs['start_date_local'].apply( lambda x: x.timestamp())
    runs['start_time_str'] = runs['start_time'].apply(lambda x: x.strftime('%H:%M:%S'))
    runs['start_time_hr_str'] = runs['start_time_str'].str[:2]
    runs['start_time_hr_int'] = runs['start_time_hr_str'].apply(lambda x: int(x))
    runs['moving_time_formatted'] = runs['moving_time'].apply(lambda x: f"{int(x/60):02d}:{int(x%60):02d}")
    return runs


def get_mileage_report_data():
    runs = get_activities()

    for column in runs.columns:
        if runs[column].dtype == 'int64':
            runs[column] = runs[column].astype(int)


    # Create the figure and axis for distance by week
    fig14, ax14 = plt.subplots(figsize=(10, 6))

    # Group the runs by week and calculate the total distance
    distance_by_week = runs.groupby(pd.Grouper(
        key='start_date_local', freq='W'))['distance'].sum()

    # Round the values to the nearest 10th
    distance_by_week = distance_by_week.round(1)
    last_week_actual = distance_by_week.values[-2]
    next_week_goal = distance_by_week.values[-2] * 1.1
    week_prog = distance_by_week.values[-1]
    miles_left = next_week_goal - week_prog

    ax14.bar(distance_by_week.index[distance_by_week.count(
    )-1], next_week_goal, color=(252/255, 76/255, 2/255), width=3.5, label='Goal')
    ax14.text(distance_by_week.index[distance_by_week.count(
    )-1], next_week_goal, last_week_actual, ha='center', color=(252/255, 76/255, 2/255), fontsize=15, fontweight='bold')


    # Plot the bar chart and add labels
    for i, val in enumerate(distance_by_week.index):
        ax14.bar(val, distance_by_week.values[i], width=5, color=(
            27/255, 117/255, 187/255))
    # for i, val in enumerate(range(1, len(distance_by_week)+1)):
        # ax14.bar(i+1, distance_by_week.values[i], width=0.8, color='blue')
        label = val  # Format date as first day of week
        # Adjust y-position of label
        ax14.text(val, distance_by_week.values[i] - 1,
                distance_by_week.values[i].round(1), ha='center', color='white', va='top', fontsize=12)

    # Set the title and axis labels
    ax14.set_title('Total Distance by Week', fontsize=24)
    ax14.set_xlabel('Week', fontsize=10)
    ax14.set_ylabel('Distance (miles)', fontsize=18)


    # Set y-axis limit with a buffer of 10%
    ax14.set_ylim(top=distance_by_week.max()*1.3)

    # Set the tick labels to be the start date of the week
    fig14.autofmt_xdate(rotation=45)

    # Set the maximum number of x-axis ticks to 10
    ax14.xaxis.set_major_locator(ticker.MultipleLocator(7))

    title = ax14.get_title()
    unique_filename = f"{uuid.uuid4()}.png"
    plt.savefig(f"visualizations/{unique_filename}")
    total_distance_by_week_plot = unique_filename

    # Create the figure and axis
    fig16, ax16 = plt.subplots(figsize=(10, 6))

    now = pd.Timestamp.now().tz_localize(pytz.utc).tz_convert(None)

    # Get the data for the last x days amd set the index to the start date
    last_7_days = runs[runs['start_date_local'].dt.tz_convert(None) >= now - pd.Timedelta(days=7)]
    last_7_days.set_index('start_date_local', inplace=True)
   
    last_14_days = runs[runs['start_date_local'].dt.tz_convert(
        None) >= now - pd.Timedelta(days=14)]
    last_14_days.set_index('start_date_local', inplace=True)
   
    last_3_days = runs[runs['start_date_local'].dt.tz_convert(
        None) >= now - pd.Timedelta(days=3)]
    last_3_days.set_index('start_date_local', inplace=True)


    # Group the runs by day and calculate the total distance
    distance_by_day = last_14_days.groupby(last_14_days.index.date)['distance'].sum()
    moving_time_by_day = last_14_days.groupby(last_14_days.index.date)['moving_time'].sum()
    moving_time_by_day_last_3_days = last_3_days.groupby(
        last_3_days.index.date)['moving_time'].sum()
    moving_time_by_day_last_7_days = last_7_days.groupby(
        last_7_days.index.date)['moving_time'].sum()
    moving_time_by_day_last_14_days = last_14_days.groupby(
        last_14_days.index.date)['moving_time'].sum()

    # Create a list of colors for the bars based on whether there is any distance or not
    colors = ['grey' if d != 0 else (27/255, 117/255, 187/255) for d in distance_by_day]

    # Plot the bar chart and add labels
    colors = [(27/255, 117/255, 187/255) if x > 0 else (240/255, 240/255, 240/255)
            for x in moving_time_by_day_last_3_days.values]
    ax16.bar(moving_time_by_day.index,
            moving_time_by_day, width=0.8, color=colors)


    # Set the title and axis labels
    ax16.set_title('Moving Time by Day', fontsize=18)
    ax16.set_xlabel('Date', fontsize=14)
    ax16.set_ylabel('Time (seconds)', fontsize=14)
    ax16.xaxis.set_major_locator(ticker.MultipleLocator())

    # Set the tick labels to be the start date of the week
    fig16.autofmt_xdate(rotation=45)

    # Save the figure
    title = ax16.get_title()
    unique_filename = f"{uuid.uuid4()}.png"
    plt.savefig(f"visualizations/{unique_filename}")
    moving_time_by_day_plot = unique_filename

    # count the number of days last week where the distance was 0
    days_zero_last_7 = int (7 - moving_time_by_day_last_7_days.count())
    days_zero_last_14 = int (14 - moving_time_by_day_last_14_days.count())
    days_zero_last_3 = int (3 - moving_time_by_day_last_3_days.count())

    y = np.asarray(runs.cadence)
    last_run_average_cadence = y[0]



    return {
        'week_prog': week_prog,
        'next_week_goal': next_week_goal,
        'miles_left': miles_left,
        'days_zero_last_3': days_zero_last_3,
        'days_zero_last_14': days_zero_last_14,
        'days_zero_last_7': days_zero_last_7,
        'last_run_average_cadence': last_run_average_cadence,
        'total_distance_by_week_plot': total_distance_by_week_plot,
        'moving_time_by_day_plot': moving_time_by_day_plot
        }


def get_cadence_report_data():
    runs = get_activities()
    # Calculate the required values for the cadence report component
    # ...
    average_cadence = ...
    pace_vs_cadence_r2 = ...
    heartrate_vs_cadence_r2 = ...

    return {
        'average_cadence': average_cadence,
        'pace_vs_cadence_r2': pace_vs_cadence_r2,
        'heartrate_vs_cadence_r2': heartrate_vs_cadence_r2
    }
