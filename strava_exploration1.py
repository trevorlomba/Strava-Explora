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
import matplotlib.ticker as ticker
# Numpy will help us handle some work with arrays.
import numpy as np
# Datetime will allow Python to recognize dates as dates, not strings.
from datetime import datetime
import strava_cleaning
import strava_api

runs = strava_cleaning.runs

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

# Create the tenth plot
fig10 = plt.figure(figsize=(10, 6))  # create overall container
fig10.subplots_adjust(bottom=0.2, left=0.1)  # adjust padding
ax10 = fig10.add_subplot(111)  # add a 1 by 1 plot to the figure
x = np.asarray(runs.start_date_local)  # convert data to numpy array
y = np.asarray(runs.cadence)
ax10.plot_date(x, y)  # plot data points in scatter plot on ax1
ax10.set_title('Average Cadence over Time')
ax10.set_xlabel('Date')
ax10.set_ylabel('Cadence (steps/min)')
# add trend line
x2 = mdates.date2num(x)
z = np.polyfit(x2, y, 1)
p = np.poly1d(z)
plt.plot(x, p(x2), 'r--')
# format the figure and display
fig10.autofmt_xdate(rotation=45)

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
fig8.subplots_adjust(bottom=0.2, left=0.2)
ax8 = fig8.add_subplot(111)  # add a subplot to the figure
x = np.asarray(runs.start_date_local)  # extract the start date of each run
y = np.asarray(runs.pace)  # extract the pace of each run
ax8.plot_date(x, y)  # plot the pace versus the start date
ax8.set_title('Pace over Time')  # set the title of the plot
ax8.set_xlabel('Date')  # set the x-axis label
ax8.set_ylabel('Pace (min/mile)')  # set the y-axis label
x2 = mdates.date2num(x)  # convert the start date to a numerical format
z = np.polyfit(x2, y, 1)  # fit a linear trendline to the data
# create a polynomial function from the trendline coefficients
p = np.poly1d(z)
plt.plot(x, p(x2), 'r--')  # plot the trendline in red dashes
fig8.autofmt_xdate(rotation=45)  # rotate the x-axis labels for readability

# Create the ninth plot
fig9 = plt.figure()  # create a new figure
fig9.subplots_adjust(bottom=0.2, left=0.2)
ax9 = fig9.add_subplot(111)  # add a subplot to the figure
fig9.autofmt_xdate(rotation=45)  # rotate the x-axis labels for readability
# extract runs that occurred in March
march_runs = runs[runs['start_date_local'].dt.month == 3]
sns.regplot(x="start_time_hr_int", y="max_heartrate", data=march_runs).set_title(
    "Max Heart Rate vs Start Time (March)")  # plot the maximum heart rate versus the start time for March runs
ax9.set_xlabel('Start Time (hour)')  # set the x-axis label
ax9.set_ylabel('Max Heart Rate')  # set the y-axis label
# print information about the run with the latest start time
print(runs.loc[runs['start_time_unix'] == runs['start_time_unix'].max()])
# print information about the run with the latest start time string
print(runs.loc[runs['start_time_str'] == runs['start_time_str'].max()])
# print information about the run with the latest start time object
print(runs.loc[runs['start_time'] == runs['start_time'].max()])
# sort the runs by the start date
sorted_runs = runs.sort_values(by=['start_date_local'])

# Create a new column for the week number
runs['week_number'] = runs['start_date_local'].dt.isocalendar().week

# Count the number of runs per week, starting on Mondays
runs_per_week = runs.groupby(['start_date_local', 'week_number']).count()
runs_per_week = runs_per_week['upload_id'].groupby('week_number').sum()


# Create the eleventh plot
fig11, ax11 = plt.subplots(figsize=(10, 6))
ax11.plot(runs_per_week.index, runs_per_week.values)
ax11.set_title('Number of Runs per Week in 2023')
ax11.set_xlabel('Week #')
ax11.set_ylabel('Number of Runs')
fig11.autofmt_xdate(rotation=45)

# Create the figure and axis for the first plot
fig12, ax12 = plt.subplots(figsize=(10, 6))

# Plot the first bar chart
runs.groupby('weekday').count()['moving_time'].plot.bar(ax=ax12)
ax12.set_title('Number of Runs by Weekday')
ax12.set_xlabel('Weekday')
ax12.set_ylabel('Number of Runs')

# Create the figure and axis for the second plot
fig13, ax13 = plt.subplots(figsize=(10, 6))

# Plot the second bar chart
runs.groupby('weekday').mean()['moving_time'].plot.bar(ax=ax13)
ax13.set_title('Average Moving Time by Weekday')
ax13.set_xlabel('Weekday')
ax13.set_ylabel('Moving Time (minutes)')

# Create the figure and axis
fig14, ax14 = plt.subplots(figsize=(10, 6))

# Group the runs by week and calculate the total distance
distance_by_week = runs.groupby(pd.Grouper(
    key='start_date_local', freq='W'))['distance'].sum()


# Round the values to the nearest 10th
distance_by_week = distance_by_week

# Calculate the moving average of the last two weeks, excluding the current week
moving_avg = (distance_by_week[distance_by_week.count()-2] * 1.1)
print (moving_avg)
print(moving_avg * 1.1)
# moving_avg = distance_by_week.rolling(window=2).apply(
#     lambda x: x.head(1).mean()).round()

# Calculate the value for the ticked line as 10% above the moving average
ticked_line = moving_avg * 1.1

ax14.bar(distance_by_week.index[distance_by_week.count(
)-1], moving_avg.round(2), color='green', width=3.5, label='Goal')
ax14.text(distance_by_week.index[distance_by_week.count(
)-1], moving_avg.round(2) + 1, moving_avg.round(2), ha='center', color='green', fontsize=13)


# Plot the bar chart and add labels
for i, val in enumerate(distance_by_week.index):
    ax14.bar(val, distance_by_week.values[i], width=5, color='blue')
# for i, val in enumerate(range(1, len(distance_by_week)+1)):
    # ax14.bar(i+1, distance_by_week.values[i], width=0.8, color='blue')
    label = val  # Format date as first day of week
    # Adjust y-position of label
    ax14.text(val, distance_by_week.values[i] -1,
              distance_by_week.values[i].round(2), ha='center', color='white', va='top', fontsize=12)

# Plot the ticked line
# ax14.axhline(y=ticked_line, linestyle='--', color='red', label='10% increase from moving average')

# Set the title and axis labels
ax14.set_title('Total Distance by Week', fontsize=24)
ax14.set_xlabel('Week', fontsize=10)
ax14.set_ylabel('Distance (miles)', fontsize=18)


# Set y-axis limit with a buffer of 10%
ax14.set_ylim(top=distance_by_week.max()*1.3)
# ax14.set_xlim(distance_by_week.values[0], distance_by_week.values[-2])

# Set the tick labels to be the start date of the week
fig14.autofmt_xdate(rotation=45)

# Set the maximum number of x-axis ticks to 10
ax14.xaxis.set_major_locator(ticker.MaxNLocator(10))

# Add a legend
# ax14.legend()

# Create the figure and axis
fig15, ax15 = plt.subplots(figsize=(10, 6))

# Group the runs by month and calculate the total distance
distance_by_month = runs.groupby(pd.Grouper(key='start_date_local', freq='M'))[
    'distance'].sum()

# Round the values to the nearest 10th
distance_by_month = distance_by_month.round(1)

# Calculate the moving average of the last two months, excluding the current month
moving_avg = distance_by_month.rolling(window=2).apply(
    lambda x: x.head(1)).mean().round(1)

# Calculate the value for the ticked line as 10% above the moving average
ticked_line = distance_by_month[1] * 1.1

# Plot the bar chart and add labels
for i, val in enumerate(distance_by_month.values):
    ax15.bar(distance_by_month.index[i], val, width=10)
    ax15.text(i, val+5, str(val), ha='center')

# Plot the ticked line
ax15.axhline(y=ticked_line, linestyle='--', color='red',
             label='10% increase from moving average')

# Set the title and axis labels
ax15.set_title('Total Distance by Month', fontsize=18)
ax15.set_xlabel('Month', fontsize=18)
ax15.set_ylabel('Distance (meters)', fontsize=18)

# Add a legend
# ax15.legend()

# Set y-axis limit with a buffer of 10%
ax15.set_ylim(top=distance_by_month.max()*1.2)

plt.show()  # display the plots in the figure
