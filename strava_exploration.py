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

print (activities['start_date_local'])

# Create new dataframe with only columns I care about
cols = ['name',
        'average_speed', 'suffer_score', 'upload_id', 'type', 'distance', 'moving_time', 'max_speed', 'total_elevation_gain',
        'start_date_local', 'average_heartrate', 'max_heartrate', 'workout_type', 'elapsed_time', 'average_cadence'
        ]
activities = activities[cols]  # Break date into start time and date
activities['start_date_local'] = pd.to_datetime(activities['start_date_local'])
activities['start_time'] = activities['start_date_local'].dt.time


# activities['start_time'] = activities['start_date_local'].dt.time.apply(
    # lambda x: x.strftime('%H:%M:%S'))
activities['start_date_local'] = activities['start_date_local'].dt.date
activities.head(5)

print(activities.head(5))

activities['start_date_local'] = pd.to_datetime(activities['start_date_local'])
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



# runs['pace'] = 60 / runs['average_speed']
# runs.loc[:, 'pace'] = 60 / runs.loc[:, 'average_speed']

# # Convert the average speed to pace in minutes per mile
# runs['pace'] = 60 / runs['average_speed']
# # Get the whole number of minutes
# runs['pace'] = runs['pace'].apply(lambda x: int(x))
# runs['pace_seconds'] = runs['pace'].apply(
#     lambda x: (x - int(x)) * 60)  # Get the decimal as seconds
# runs['pace'] = runs['pace'].apply(
#     lambda x: f"{int(x)}:{int((x - int(x)) * 60):02d}")  # Format as MM:SS
# runs['pace'] = pd.to_numeric(runs['pace'])


print(runs.count(1))


# print(runs_2023)

# Plot the data
sns.set(style="ticks", context="talk")

# Create a first plot
fig = plt.figure(figsize=(10, 6))  # create overall container
fig.subplots_adjust(bottom=0.2, left=0.1)  # adjust padding
ax1 = fig.add_subplot(111)  # add a 1 by 1 plot to the figure
x = np.asarray(runs.start_date_local)  # convert data to numpy array
y = np.asarray(runs.pace)
ax1.plot_date(x, y)  # plot data points in scatter plot on ax1
ax1.set_title('Average Pace over Time')
ax1.set_xlabel('Date')
ax1.set_ylabel('Pace (min/mile)')

# add trend line
x2 = mdates.date2num(x)
z = np.polyfit(x2, y, 1)
p = np.poly1d(z)
plt.plot(x, p(x2), 'r--')

# format the figure and display
fig.autofmt_xdate(rotation=45)

# Create a second plot
fig2 = plt.figure(figsize=(10, 6))
fig2.subplots_adjust(bottom=0.2, left=0.1)
ax2 = fig2.add_subplot(111)
sns.regplot(x='pace', y='cadence', data=runs).set_title("Pace vs Cadence")
ax2.set_xlabel('Pace (min/mile)')
ax2.set_ylabel('Cadence (steps/min)')

# Create a third plot
fig3 = plt.figure(figsize=(10, 6))
fig3.subplots_adjust(bottom=0.2, left=0.1)
ax3 = fig3.add_subplot(111)
sns.regplot(x='average_heartrate', y='cadence', data=runs).set_title(
    "Average Heart Rate vs Cadence")
ax3.set_xlabel('Average Heart Rate')
ax3.set_ylabel('Cadence (steps/min)')

# Create a fourth plot
fig4 = plt.figure(figsize=(10, 6))
fig4.subplots_adjust(bottom=0.2, left=0.1)
ax4 = fig4.add_subplot(111)
sns.regplot(x='distance', y='max_heartrate', data=runs).set_title(
    "Max Heart Rate vs Distance")
ax4.set_xlabel('Distance (miles)')
ax4.set_ylabel('Max Heart Rate')

# Create a fifth plot
fig5 = plt.figure(figsize=(10, 6))
fig5.subplots_adjust(bottom=0.2, left=0.1)
ax5 = fig5.add_subplot(111)
sns.regplot(x='distance', y='suffer_score', data=runs).set_title(
    "Suffer Score vs Distance")
ax5.set_xlabel('Distance (miles)')
ax5.set_ylabel('Suffer Score')

# Create a sixth plot
fig6 = plt.figure(figsize=(10, 6))
fig6.subplots_adjust(bottom=0.2, left=0.1)
ax6 = fig6.add_subplot(111)
sns.regplot(y='max_heartrate', x='suffer_score', data=runs).set_title(
    "Max Heart Rate vs Suffer Score")
ax6.set_xlabel('Suffer Score')
ax6.set_ylabel('Max Heart Rate')

# Create a seventh plot
fig7 = plt.figure(figsize=(10, 6))
fig7.subplots_adjust(bottom=0.2, left=0.1)
ax7 = fig7.add_subplot(111)
march_runs = runs[runs['start_date_local'].dt.month == 3]
sns.regplot(x="start_time_hr_int", y="max_heartrate", data=march_runs).set_title(
    "Max Heart Rate vs Start Time (March)")
ax7.set_xlabel('Start Time (hour of day)')
ax7.set_ylabel('Max Heart Rate')

# Create the eighth plot
fig8 = plt.figure()  # create a new figure
ax8 = fig8.add_subplot(111)  # add a subplot to the figure
x = np.asarray(runs.start_date_local)  # extract the start date of each run
y = np.asarray(runs.pace)  # extract the pace of each run
ax8.plot_date(x, y)  # plot the pace versus the start date
ax8.set_title('Average Cadence over Time')  # set the title of the plot
x2 = mdates.date2num(x)  # convert the start date to a numerical format
z = np.polyfit(x2, y, 1)  # fit a linear trendline to the data
# create a polynomial function from the trendline coefficients
p = np.poly1d(z)
plt.plot(x, p(x2), 'r--')  # plot the trendline in red dashes
fig8.autofmt_xdate(rotation=45)  # rotate the x-axis labels for readability

# Create the ninth plot
fig9 = plt.figure()  # create a new figure
ax9 = fig9.add_subplot(111)  # add a subplot to the figure
fig9.autofmt_xdate(rotation=45)  # rotate the x-axis labels for readability
# extract runs that occurred in March
march_runs = runs[runs['start_date_local'].dt.month == 3]
sns.regplot(x="start_time_hr_int", y="max_heartrate", data=march_runs).set_title(
    "Max Heart Rate vs Start Time (March)")  # plot the maximum heart rate versus the start time for March runs
# print information about the run with the latest start time
print(runs.loc[runs['start_time_unix'] == runs['start_time_unix'].max()])
# print information about the run with the latest start time string
print(runs.loc[runs['start_time_str'] == runs['start_time_str'].max()])
# print information about the run with the latest start time object
print(runs.loc[runs['start_time'] == runs['start_time'].max()])
# sort the runs by the start date
sorted_runs = runs.sort_values(by=['start_date_local'])
plt.show()  # display the plots in the figure


print(plot)
