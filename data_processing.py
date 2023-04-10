import numpy as np
import statsmodels.api as sm
import uuid
import seaborn as sns
import pandas as pd
from pandas import json_normalize
import strava_api
import strava_cleaning
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import pytz
from datetime import timedelta

def get_activities():
    activities = json_normalize(strava_api.my_dataset)

    # Create new dataframe with only columns I care about
    cols = ['name',        'average_speed', 'suffer_score', 'upload_id', 'type', 'distance', 'moving_time', 'max_speed',
            'total_elevation_gain',        'start_date_local', 'average_heartrate', 'max_heartrate', 'workout_type', 'elapsed_time', 'average_cadence']
    activities = activities[cols]

    # Break date into start time and date
    activities['start_date_local'] = pd.to_datetime(activities['start_date_local'])
    activities['weekday'] = activities['start_date_local'].map(lambda x: x.weekday)
    activities['start_time'] = activities['start_date_local'].dt.time

    runs = activities.loc[(activities['type'] == 'Run') & (
        activities['start_date_local'].dt.year == 2023)].copy()
    runs.loc[:, 'distance'] = runs['distance'] * 0.000621371
    runs.loc[:, 'pace'] = (runs['moving_time'] / 60) / (runs['distance'])
    runs.loc[:, 'cadence'] = (runs['average_cadence'] * 2)
    # runs["start_time"] = pd.to_datetime(runs["start_time"])
    runs.loc[:, 'start_time_unix'] = runs['start_date_local'].apply(
        lambda x: x.timestamp())
    runs.loc[:, 'start_time_unix'] = runs['start_date_local'].apply(
        lambda x: x.timestamp())
    runs.loc[:, 'start_time_str'] = runs['start_time'].apply(
        lambda x: x.strftime('%H:%M:%S'))
    runs.loc[:, 'start_time_hr_str'] = runs['start_time_str'].str[:2]
    runs.loc[:, 'start_time_hr_int'] = runs['start_time_hr_str'].apply(
        lambda x: int(x))
    runs.loc[:, 'moving_time_formatted'] = runs['moving_time'].apply(
        lambda x: f"{int(x/60):02d}:{int(x%60):02d}")

    return runs


def get_mileage_report_data():
    runs = get_activities()


    for column in runs.columns:
        if runs[column].dtype == 'int64':
            runs.loc[:, column] = runs.loc[:, column].astype(int)


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

  # First goal bar
    ax14.bar(distance_by_week.index[distance_by_week.count()-1], next_week_goal, color=(252/255, 76/255, 2/255), width=3.5, label='Goal 1')
    ax14.text(distance_by_week.index[distance_by_week.count()-1], next_week_goal * 1.05, f'{next_week_goal:.2f}', ha='center', color=(252/255, 76/255, 2/255), fontsize=15, fontweight='bold')

    # Second goal bar (10% greater than the first goal)
    next_week_goal_2 = next_week_goal * 1.1
    ax14.bar(distance_by_week.index[distance_by_week.count()-1] + timedelta(weeks=1), next_week_goal_2, color=(252/255, 76/255, 2/255), width=3.5, label='Goal 2', alpha=0.3)
    ax14.text(distance_by_week.index[distance_by_week.count()-1] + timedelta(weeks=1), next_week_goal_2 * 1.05,
              f'{next_week_goal_2:.2f}', ha='center', color=(252/255, 76/255, 2/255), fontsize=15, fontweight='bold', alpha=0.6)

    # Third goal bar (10% greater than the second goal)
    next_week_goal_3 = next_week_goal_2 * 1.1
    ax14.bar(distance_by_week.index[distance_by_week.count()-1] + timedelta(weeks=2), next_week_goal_3, color=(252/255, 76/255, 2/255), width=3.5, label='Goal 3', alpha=0.3)
    ax14.text(distance_by_week.index[distance_by_week.count()-1] + timedelta(weeks=2), next_week_goal_3 * 1.05,
              f'{next_week_goal_3:.2f}', ha='center', color=(252/255, 76/255, 2/255), fontsize=15, fontweight='bold', alpha=0.6)



    # Plot the bar chart and add labels
    for i, val in enumerate(distance_by_week.index):
        ax14.bar(val, distance_by_week.values[i], width=5, color=(
            27/255, 117/255, 187/255))
        label = val  # Format date as first day of week
        # Adjust y-position of label
        ax14.text(val, distance_by_week.values[i] - 1,
                distance_by_week.values[i].round(1), fontweight='bold', fontsize=11, ha='center', color='white', va='top')

    # Set the title and axis labels
    ax14.set_title('Total Distance by Week', fontsize=24)
    ax14.set_xlabel('Week', fontsize=10)
    ax14.set_ylabel('Distance (miles)', fontsize=18)


    # Set y-axis limit with a buffer of 10%
    ax14.set_ylim(top=next_week_goal_3 * 1.2)

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
    last_7_days = runs[runs['start_date_local'].dt.tz_convert(
        None) >= now - pd.Timedelta(days=6)].copy()
    last_7_days.set_index('start_date_local', inplace=True)

    last_14_days = runs[runs['start_date_local'].dt.tz_convert(
        None) >= now - pd.Timedelta(days=13)].copy()
    last_14_days.set_index('start_date_local', inplace=True)
    last_3_days = runs[runs['start_date_local'].dt.tz_convert(
        None) >= now - pd.Timedelta(days=2)].copy()
    last_3_days.set_index('start_date_local', inplace=True)


    # Group the runs by day and calculate the total distance
    distance_by_day = last_14_days.groupby(
        last_14_days.index.date)['distance'].sum()
    moving_time_by_day = last_14_days.groupby(last_14_days.index.date)[
        'moving_time'].sum()
    moving_time_by_day_last_3_days = last_3_days.groupby(
        last_3_days.index.date)['moving_time'].sum()
    moving_time_by_day_last_7_days = last_7_days.groupby(
        last_7_days.index.date)['moving_time'].sum()
    moving_time_by_day_last_14_days = last_14_days.groupby(
        last_14_days.index.date)['moving_time'].sum()

    # Create a list of colors for the bars based on whether there is any distance or not
    colors = ['grey' if d != 0 else (27/255, 117/255, 187/255)
            for d in distance_by_day]

    # Plot the bar chart and add labels
    colors = [(27/255, 117/255, 187/255) if x > 0 else (240/255, 240/255, 240/255)
            for x in moving_time_by_day_last_3_days.values]
    ax16.bar(moving_time_by_day.index,
            moving_time_by_day, width=0.8, color=colors)

    #set y label to be minutes:seconds with both second digits
    ax16.set_yticklabels([f"{int(x/60)}:{int(x%60):02d}" for x in ax16.get_yticks()])

    # ax16.set_yticklabels([str(int(x/60)) + ':' + str(int(x % 60)) for x in ax16.get_yticks()])
    # ax16.set_yticklabels([str(int(x/60)) + ':' + str(int(x % 60)) for x in ax16.get_yticks()])

    # Set the title and axis labels
    # ax14.set_title('Total Distance by Week', fontsize=24)
    ax16.set_title('Moving Time by Day', fontsize=24)
    ax16.set_xlabel('Date', fontsize=14)
    ax16.set_ylabel('Time (minutes)', fontsize=14)
    ax16.xaxis.set_major_locator(ticker.MultipleLocator())


    # Set the tick labels to be the start date of the week
    fig16.autofmt_xdate(rotation=45)

    # Save the figure
    title = ax16.get_title()
    unique_filename = f"{uuid.uuid4()}.png"
    plt.savefig(f"visualizations/{unique_filename}")
    moving_time_by_day_plot = unique_filename

    # count the number of days last week where the distance was 0
    days_zero_last_7 = int(7 - moving_time_by_day_last_7_days.count())
    days_zero_last_14 = int(14 - moving_time_by_day_last_14_days.count())
    days_zero_last_3 = int(3 - moving_time_by_day_last_3_days.count())

    #print moving days last 14 days
    print('moving_time_by_day_last_14_days')
    print(moving_time_by_day_last_14_days)

    y = np.asarray(runs.cadence)
    last_run_average_cadence = y[0]


    return {
        'week_prog': week_prog.round(2),
        'next_week_goal': next_week_goal.round(2),
        'miles_left': miles_left.round(2),
        'days_zero_last_3': days_zero_last_3,
        'days_zero_last_14': days_zero_last_14,
        'days_zero_last_7': days_zero_last_7,
        'last_run_average_cadence': last_run_average_cadence.round(2),
        'total_distance_by_week_plot': total_distance_by_week_plot,
        'moving_time_by_day_plot': moving_time_by_day_plot
        }


def get_cadence_report_data():
    runs = get_activities()
    # create the seventeenth plot
    fig17 = plt.figure(figsize=(10, 6))
    fig17.subplots_adjust(bottom=0.2, left=0.1)
    ax17 = fig17.add_subplot(111)
    sns.regplot(x='pace', y='cadence', data=runs, ax=ax17, color=(
        27/255, 117/255, 187/255)).set_title(
        "Average Pace vs Average Cadence", fontsize=26)
    ax17.set_xlabel('Average Pace (min/mile)',
                    fontsize=18, labelpad=18, ha='center')
    ax17.set_ylabel('Average Cadence (spm)', fontsize=18, labelpad=18, va='center')

    # add R-squared value if significant
    model = sm.OLS.from_formula('cadence ~ pace', data=runs)
    results = model.fit()
    r_squared = results.rsquared
    pace_vs_cadence_r2 = r_squared
    if r_squared > 0.1:
        ax17.annotate(f"R-squared = {r_squared:.2f}",
                    xy=(0.5, 0.9), xycoords='axes fraction')

    title = ax17.get_title()
    unique_filename = f"{uuid.uuid4()}.png"
    plt.savefig(f"visualizations/{unique_filename}")
    average_pace_vs_average_cadence_plot = unique_filename


    # create the eighteenth plot
    fig18 = plt.figure(figsize=(10, 6))
    fig18.subplots_adjust(bottom=0.2, left=0.1)
    ax18 = fig18.add_subplot(111)
    sns.regplot(x='cadence', y='average_heartrate', data=runs, ax=ax18, color=(
        27/255, 117/255, 187/255)).set_title(
        "Average Heartrate vs Average Cadence", fontsize=26)
    # set font size and alignment of x and y labels
    
    ax18.set_xlabel('Average Heartrate (bpm)',
                    fontsize=18, labelpad=18, ha='center')
    ax18.set_ylabel('Average Cadence (spm)', fontsize=18,
                    labelpad=18,  va='center')

    # add R-squared value if significant
    model = sm.OLS.from_formula('average_heartrate ~ cadence', data=runs)
    results = model.fit()
    r_squared = results.rsquared
    heartrate_vs_cadence_r2 = r_squared
    if r_squared > 0.1:
        ax18.annotate(f"R-squared = {r_squared:.2f}",
                    xy=(0.9, 0.9), xycoords='axes fraction')

    title = ax18.get_title()
    unique_filename = f"{uuid.uuid4()}.png"
    plt.savefig(f"visualizations/{unique_filename}")
    average_heart_rate_vs_average_cadence_plot = unique_filename

    runs.loc[:, 'distance'] = runs['distance'] * 0.000621371
    average_cadence = runs['cadence'].mean()
    pace_vs_cadence_r2 = pace_vs_cadence_r2
    heartrate_vs_cadence_r2 = heartrate_vs_cadence_r2
    recent_run = runs.head(1)
    print(recent_run)
    most_recent_cadence = recent_run['cadence'].values[0]
    print(most_recent_cadence)

    return {
        'average_cadence': average_cadence.round(2),
        'pace_vs_cadence_r2': pace_vs_cadence_r2.round(2),
        'heartrate_vs_cadence_r2': heartrate_vs_cadence_r2.round(2),
        'average_heart_rate_vs_average_cadence_plot': average_heart_rate_vs_average_cadence_plot,
        'average_pace_vs_average_cadence_plot': average_pace_vs_average_cadence_plot,
        'most_recent_cadence': most_recent_cadence,
    }
