import jinja2
import os
from strava_exploration1 import next_week_goal, week_prog, miles_left, days_zero_last_14, days_zero_last_3, days_zero_last_7

print('hello')

# Load the template from a file
template_loader = jinja2.FileSystemLoader(searchpath='./templates')
template_env = jinja2.Environment(loader=template_loader)
template = template_env.get_template('report.html')

# Render the template
html_output = template.render(next_week_goal=next_week_goal.round(1), week_prog=week_prog.round(1), miles_left=miles_left.round(1),  total_distance_by_week='visualizations/Total Distance by Week.png',
                              average_cadence_over_time='visualizations/Average Cadence over Time.png',
                              number_of_runs_per_week='visualizations/Number of Runs per Week in 2023.png',
                              days_zero_last_14=days_zero_last_14, days_zero_last_3=days_zero_last_3, days_zero_last_7_days=days_zero_last_7
                              )

print(os.getcwd())  # print current working directory

# Print the HTML output (for testing purposes)
print(html_output)

# Save the HTML output to a file
with open('report.html', 'w') as f:
    f.write(html_output)
