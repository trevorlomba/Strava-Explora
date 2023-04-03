# Strava-Explora

This project utilizes the Strava API to pull information about runs and analyze them using Python. The data is manipulated and visualized using Pandas, Seaborn, Matplotlib, Numpy, and Datetime libraries.
Installation

    Create an account on Strava and go to https://www.strava.com/settings/api to obtain your API key.

    Clone or download this repository.

    Install the required libraries using the command: pip install -r requirements.txt.

    Change the following fields in the strava_api.py file to your own Strava API credentials: client_id, client_secret, refresh_token.

    Run the strava_api.py script.

    Run the run_analysis.py script to analyze your Strava runs.

## Usage

The run_analysis.py script retrieves all of your runs for the current year and creates a dataframe with the following columns: name, average_speed, suffer_score, upload_id, type, distance, moving_time, max_speed, total_elevation_gain, start_date_local, average_heartrate, max_heartrate, workout_type, elapsed_time, and average_cadence.

The script then cleans and manipulates the data by converting the date column into a datetime object and breaking it into separate date and time columns, converting distance from meters to miles, calculating pace, and creating additional columns for plotting.

Finally, the script plots the data in multiple graphs.

## Credits

This project was created by Trevor Lomba (tjlomba95@gmail.com) as a personal project. Special thanks to Strava for providing the API and to the creators of the Python libraries used in this project.
