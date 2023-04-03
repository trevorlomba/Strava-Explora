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
plot = sns.regplot(x='distance', y='pace', data=runs).set_title( "Pace vs Distance")


fig = plt.figure()  # create overall container
ax1 = fig.add_subplot(111)  # add a 1 by 1 plot to the figure
x = np.asarray(runs.start_date_local)  # convert data to numpy array
y = np.asarray(runs.pace)
ax1.plot_date(x, y)  # plot data points in scatter plot on ax1
ax1.set_title('Average Pace over Time')
# ax1.set_ylim([0,5])
# add trend line
x2 = mdates.date2num(x)
z = np.polyfit(x2, y, 1)
p = np.poly1d(z)
plt.plot(x, p(x2), 'r--')
# format the figure and display
fig.autofmt_xdate(rotation=45)
# fig.tight_layout()

# Create a third plot
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
sns.regplot(x='pace', y='cadence', data=runs).set_title("Pace vs Cadence")

# Create a fourth plot
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
sns.regplot(x='average_heartrate', y='cadence', data=runs).set_title("Average Heart Rate vs Cadence")

# Create a fifth plot
fig4 = plt.figure()
ax4 = fig4.add_subplot(111)
sns.regplot(x='distance', y='max_heartrate', data=runs).set_title("Max Heart Rate vs Distance")

# Create a sixth plot
fig5 = plt.figure()
ax5 = fig5.add_subplot(111)
sns.regplot(x='distance', y='suffer_score', data=runs).set_title("Suffer Score vs Distance")


# Create a sixth plot
fig6 = plt.figure()
ax6 = fig6.add_subplot(111)
sns.regplot(y='max_heartrate', x='suffer_score', data=runs).set_title("Max Heart Rate vs Suffer Score")


# # Create a sixth plot
# fig7 = plt.figure()
# ax7 = fig7.add_subplot(111)
# x = np.asarray(runs.start_date_local)  # convert data to numpy array
# y = np.asarray(runs.pace)
# ax7.plot_date(x, y)  # plot data points in scatter plot on ax7
# ax7.set_title('Average Cadence over Time')
# # add trend line
# x2 = mdates.date2num(x)
# z = np.polyfit(x2, y, 1)
# p = np.poly1d(z)
# plt.plot(x, p(x2), 'r--')
# # format the figure and display
# fig7.autofmt_xdate(rotation=45)




# Create a eighth plot
fig8 = plt.figure()
ax8 = fig8.add_subplot(111)

fig8.autofmt_xdate(rotation=45)

# Filter runs at or over 3 miles
# runs_over_3_miles = runs[(runs['distance'] >= 3) & (runs['cadence'] >= 165)]
march_runs = runs[runs['start_date_local'].dt.month == 3]

# Plot with filtered data
# sns.regplot(x="start_time_hr_int", y="max_heartrate", data=runs_over_3_miles).set_title(
#     "Max Heart Rate vs Start Time (3+ Miles)")
sns.regplot(x="start_time_hr_int", y="max_heartrate", data=march_runs).set_title(
    "Max Heart Rate vs Start Time (March)")

print(runs.loc[runs['start_time_unix'] == runs['start_time_unix'].max()])
print(runs.loc[runs['start_time_str'] == runs['start_time_str'].max()])
print(runs.loc[runs['start_time'] == runs['start_time'].max()])

sorted_runs = runs.sort_values(by=['start_date_local'])
print(sorted_runs)

plt.show()

print(plot)
