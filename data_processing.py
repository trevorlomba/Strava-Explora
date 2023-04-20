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
import matplotlib.colors as mcolors
import pytz
from datetime import timedelta, datetime
import math


def get_activities():
    # Fetch data from Strava API
    strava_api.fetch_data()

    activities = json_normalize(strava_api.my_dataset)


    # Create new dataframe with only columns I care about
    cols = ['name', 'average_speed', 'suffer_score', 'upload_id', 'type', 'distance', 'moving_time', 'max_speed', 'total_elevation_gain', 'start_date_local', 'average_heartrate', 'max_heartrate', 'workout_type', 'elapsed_time', 'average_cadence']
    
    activities = activities[cols]

    # Break date into start time and date
    activities['start_date_local'] = pd.to_datetime(activities['start_date_local'])
    activities['weekday'] = activities['start_date_local'].map(lambda x: x.weekday)
    activities['start_time'] = activities['start_date_local'].dt.time

    runs = activities.loc[(activities['type'] == 'Run') & (
        activities['start_date_local'].dt.year == 2023)].copy()

    # Fill missing values with dummy values
    dummy_values = {
        'average_cadence': 170,
        'average_heartrate': 150,
        'max_heartrate': 170,
        'suffer_score': 30,
        'average_cadence': 80,
        'average_speed': 0,
        'max_speed': 0,
        'total_elevation_gain': 0,
        'distance': 4,
        'moving_time': 1700,
        'elapsed_time': 1700,
        'average_heartrate': 160,
        'max_heartrate': 170,
        'start_time': '00:00:00',
        'weekday': 2,
        'start_date_local': '2023-01-01 00:00:00',
        'upload_id': 0,
    }

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

    # print(most_recent_run)


    return runs


def get_mileage_report_data():
    runs = get_activities()

    most_recent_run = runs.iloc[0]
    most_recent_run_date = most_recent_run['start_date_local'].date()
    print(most_recent_run_date)
    most_recent_run_today = most_recent_run_date == datetime.today().date()
    print(most_recent_run_today)

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
    previous_week = distance_by_week[-2]


    week_before_previous = distance_by_week[-3] if len(
        distance_by_week) >= 3 else 0
    
    two_weeks_before_previous = distance_by_week[-4] if len(
        distance_by_week) >= 4 else 0
    
    three_weeks_before_previous = distance_by_week[-5] if len(
        distance_by_week) >= 5 else 0

    highest_value = max(previous_week, week_before_previous,
                        two_weeks_before_previous, three_weeks_before_previous)
    next_week_goal = highest_value * 1.1

    # print distances by weeks
    print('here are the distances by week')
    # print(distance_by_week)
    print('here are the distances by week')
    print(distance_by_week)
    print('here are the distances by week before previous')
    print(week_before_previous)
    print('here are the distances by week before previous')
    print(two_weeks_before_previous)
    print(three_weeks_before_previous)
    print(highest_value)

    print('here are the distances by week')
    print(distance_by_week.values[-1])
    print(distance_by_week.values)
    week_prog = distance_by_week.values[-1]
    print('here is the week prog')
    print(week_prog)
    miles_left = next_week_goal - week_prog


  # First goal bar
    ax14.bar(distance_by_week.index[distance_by_week.count()-1], next_week_goal, color=(252/255, 76/255, 2/255), width=3.5, label='Goal 1', alpha=.8)
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
            27/255, 117/255, 187/255), alpha=.8)
        label = val  # Format date as first day of week
        # Adjust y-position of label
        ax14.text(val, distance_by_week.values[i] - 1,
                distance_by_week.values[i].round(1), fontweight='bold', fontsize=11, ha='center', color='white', va='top')

    # Set the title and axis labels
    ax14.set_title('Total Distance by Week', fontsize=24)
    ax14.set_xlabel('Week', fontsize=14)
    ax14.set_ylabel('Distance (miles)', fontsize=14)


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
        None) >= now - pd.Timedelta(days=7)].copy()
    last_7_days.set_index('start_date_local', inplace=True)

    last_15_days = runs[runs['start_date_local'].dt.tz_convert(
        None) >= now - pd.Timedelta(days=15)].copy()
    last_15_days.set_index('start_date_local', inplace=True)
    last_14_days = runs[runs['start_date_local'].dt.tz_convert(
        None) >= now - pd.Timedelta(days=14)].copy()
    last_14_days.set_index('start_date_local', inplace=True)
    last_3_days = runs[runs['start_date_local'].dt.tz_convert(
        None) >= now - pd.Timedelta(days=3)].copy()
    last_3_days.set_index('start_date_local', inplace=True)

    # print(last_14_days['moving_time'])
    # print(last_7_days['moving_time'])
    # print(last_3_days['moving_time'])

    # Group the runs by day and calculate the total distance
    distance_by_day = last_14_days.groupby(
        last_14_days.index.date)['distance'].sum()
    moving_time_by_day = last_15_days.groupby(last_15_days.index.date)[
        'moving_time'].sum()
    moving_time_by_day_last_3_days = last_3_days.groupby(
        last_3_days.index.date)['moving_time'].sum()
    moving_time_by_day_last_7_days = last_7_days.groupby(
        last_7_days.index.date)['moving_time'].sum()
    moving_time_by_day_last_14_days = last_14_days.groupby(
        last_14_days.index.date)['moving_time'].sum()

    # print(moving_time_by_day_last_3_days)
    # print(moving_time_by_day_last_7_days)
    # print(moving_time_by_day_last_14_days)

    today = pd.Timestamp.now().tz_localize(pytz.utc).tz_convert(None)


    # Calculate the longest run in the last week



    def get_last_week_data(today):
        days_since_sunday = (today.isoweekday() % 7) or 7

        print('days since sunday')
        print(days_since_sunday)

        # Calculate the start and end dates for the last week
        end_date_last_week = (today - pd.DateOffset(days=days_since_sunday)).replace(hour=23, minute=59, second=59, microsecond=0)
        print('end date last week')
        print(end_date_last_week)
        start_date_last_week = end_date_last_week - pd.DateOffset(days=6)
        print('start date last week')
        print(start_date_last_week)

        return runs[(runs['start_date_local'].dt.tz_convert(None) >= start_date_last_week) &
                    (runs['start_date_local'].dt.tz_convert(None) <= end_date_last_week)].copy()


    # Change the second argument to set a different start day
    last_week_data = get_last_week_data(today)

    # Find the longest run in the last week ending on the day before the first day of this week
    longest_run_last_2_weeks = last_15_days['distance'].max()

    def days_left_in_week(today):
        today_weekday = today.isoweekday()
        days_left = 8 - today_weekday if today_weekday != 7 else 1
        return days_left


    days_left = days_left_in_week(today)
    # Calculate the number of miles left to run this week
    miles_left = next_week_goal - week_prog


    # Get today's date
    today = pd.Timestamp.now().normalize()

    # Calculate the days left in the week (days until next Monday)
    days_left = (7 - today.weekday()) % 7

    # if days_left is zero, then it is Monday, so set to 7
    if days_left == 0:
        days_left = 7


    # Find the most recent Monday
    most_recent_monday = today - pd.Timedelta(days=days_left)

    # Filter the runs since the most recent Monday including today
    runs_since_monday = runs[(runs['start_date_local'].dt.tz_convert(
        None) >= most_recent_monday) & (runs['start_date_local'].dt.tz_convert(None) <= today)]

    # Find the longest run since the most recent Monday including today
    longest_run_since_monday = runs_since_monday['distance'].max()

    print("The longest run since the most recent Monday including today is:",
        longest_run_since_monday)

    # Find the longest run since Monday including today and the most recent monday
    longest_run_since_monday = runs[(runs['start_date_local'].dt.tz_convert(None) >= today - pd.Timedelta(days=7 - days_left))]['distance'].max()

    # If there was no run since monday, set the longest run to 0
    if not longest_run_since_monday:
        longest_run_since_monday = 0

    # Check if the longest run in the last week was above last week's long run
    long_run_improved = longest_run_since_monday == longest_run_last_2_weeks

    # Calculate the average miles left to run this week
    avg_miles_left = miles_left / days_left

    # Calculate the goal long run
    goal_long_run = longest_run_last_2_weeks * 1.1

    # Days left minus this weeks long run
    if not long_run_improved:
        days_left_minus_long_run = days_left - 1
    else:
        days_left_minus_long_run = days_left

    # Calculate the miles left that week minus the long run goal
    if not long_run_improved:
        miles_left_minus_long_run_goal = miles_left - (goal_long_run)
    else:
        miles_left_minus_long_run_goal = miles_left

    # print(f"Goal mileage this week: {next_week_goal}")
    # print("Days left in the week:", days_left)
    # print("Miles left to run this week:", miles_left)
    # print("Average miles left to run this week:", avg_miles_left)
    # print("Longest run since Monday:", longest_run_since_monday)
    # print("Longest last week:", longest_run_last_2_weeks)
    # print("Long run improved:", long_run_improved)
    # print(f"Goal long run (10% more than the longest last week): {goal_long_run:.2f}")
    # print("Miles left minus long run goal:", miles_left_minus_long_run_goal)
    # print(f"Days left in the week minus the long run: {days_left_minus_long_run}")

 
    # print("\nLast week's distances:")
    # print(last_week_data['distance'])



    # Calculate a goal variable of 10% more than the longest run in the last week

    # Display the calculated variables

    # Create a list of colors for the bars based on whether there is any distance or not




    max_moving_time = moving_time_by_day.max()
    print(max_moving_time)

    min_moving_time = moving_time_by_day.min()

    for i, yval in enumerate(moving_time_by_day):
        # ax.text(i, yval + 0.01, f"{yval:.2f}", ha="center", va="bottom")
        print(yval / max_moving_time)
        ax16.bar(moving_time_by_day.index[i], yval, width=0.8, alpha=(
            yval / max_moving_time)*.7, color=(
            27/255, 117/255, 187/255))


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

    # print moving days last 14 days
    # print('moving_time_by_day_last_14_days')
    # print(moving_time_by_day_last_14_days)

    y = np.asarray(runs.cadence)
    last_run_average_cadence = y[0]

    # if no runs since monday, set longest run since monday to 0
    if longest_run_since_monday is None or math.isnan(longest_run_since_monday):
        longest_run_since_monday = float(0)
        # next_week_goal = next_week_goal * 1.1
        miles_left = next_week_goal
        avg_miles_left = miles_left / days_left
        long_run_improved = 'False'
        days_left_minus_long_run = days_left - 1
        miles_left_minus_long_run_goal = miles_left

    


    # print('week_prog:', week_prog)
    # print('next_week_goal:', next_week_goal)
    # print('miles_left:', miles_left)
    # print('days_zero_last_3:', days_zero_last_3)
    # print('days_zero_last_14:', days_zero_last_14)
    # print('days_zero_last_7:', days_zero_last_7)
    # print('last_run_average_cadence:', last_run_average_cadence)
    # print('total_distance_by_week_plot:', total_distance_by_week_plot)
    # print('moving_time_by_day_plot:', moving_time_by_day_plot)
    # print('days_left:', days_left)
    # print('avg_miles_left:', avg_miles_left)
    # print('longest_run_last_2_weeks:', longest_run_last_2_weeks)
    # print('longest_run_since_monday:', longest_run_since_monday)
    # print('goal_long_run:', goal_long_run)
    # print('long_run_improved:', long_run_improved)
    # print('miles_left_minus_long_run_goal:', miles_left_minus_long_run_goal)
    # print('days_left_minus_long_run:', days_left_minus_long_run)

    print(week_prog)


    return {
        'week_prog': week_prog,
        'next_week_goal': next_week_goal,
        'miles_left': miles_left,
        'days_zero_last_3': days_zero_last_3,
        'days_zero_last_14': days_zero_last_14,
        'days_zero_last_7': days_zero_last_7,
        'last_run_average_cadence': last_run_average_cadence,
        'total_distance_by_week_plot': total_distance_by_week_plot,
        'moving_time_by_day_plot': moving_time_by_day_plot,
        'days_left': days_left,
        'avg_miles_left': avg_miles_left,
        'longest_run_last_2_weeks': longest_run_last_2_weeks,
        'goal_long_run': goal_long_run,
        'longest_run_since_monday': longest_run_since_monday,
        'long_run_improved': str(long_run_improved),
        'miles_left_minus_long_run_goal': miles_left_minus_long_run_goal,
        'days_left_minus_long_run': days_left_minus_long_run,
        'most_recent_run_today': most_recent_run_today
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


    # Add a new column for average heartrate per mile
    runs['average_heartrate_per_mile'] = runs['average_heartrate'] / runs['distance']

    # Create the eighteenth plot
    fig18 = plt.figure(figsize=(10, 6))
    fig18.subplots_adjust(bottom=0.2, left=0.1)
    ax18 = fig18.add_subplot(111)
    sns.regplot(x='cadence', y='average_heartrate_per_mile', data=runs, ax=ax18, color=(
        27/255, 117/255, 187/255)).set_title(
        "Average Heartrate per Mile vs Average Cadence", fontsize=26)

    # Set font size and alignment of x and y labels
    ax18.set_xlabel('Average Heartrate per Mile (bpm)',
                    fontsize=18, labelpad=18, ha='center')
    ax18.set_ylabel('Average Cadence (spm)', fontsize=18,
                    labelpad=18,  va='center')

    # Add R-squared value if significant
    model = sm.OLS.from_formula('average_heartrate_per_mile ~ cadence', data=runs)
    results = model.fit()
    r_squared = results.rsquared
    heartrate_vs_cadence_r2 = r_squared
    if r_squared > 0.1:
        ax18.annotate(f"R-squared = {r_squared:.2f}",
                    xy=(0.9, 0.9), xycoords='axes fraction')

    title = ax18.get_title()
    unique_filename = f"{uuid.uuid4()}.png"
    plt.savefig(f"visualizations/{unique_filename}")
    average_heart_rate_per_mile_vs_average_cadence_plot = unique_filename

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
        'average_heart_rate_vs_average_cadence_plot': average_heart_rate_per_mile_vs_average_cadence_plot,
        'average_pace_vs_average_cadence_plot': average_pace_vs_average_cadence_plot,
        'most_recent_cadence': most_recent_cadence,
    }
