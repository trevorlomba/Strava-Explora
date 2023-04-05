# Strava-Explora

This project utilizes the Strava API to pull information about runs and analyze them using Python. The data is manipulated and visualized using Pandas, Seaborn, Matplotlib, Numpy, and Datetime libraries.

As I collect more data over time, my plan is to expand on this project by incorporating additional data points such as sleep, stress, and recovery data from my Garmin watch. With these additional data points, I will have a more comprehensive understanding of the factors that contribute to my overall health and fitness.

Using this data, my goal is to create a model that will allow me to gradually ramp up my mileage and effort over time in order to reduce the risk of injury. By analyzing the patterns and trends in my data, I will be able to determine the optimal amount of mileage and effort for each week, based on my current fitness level and the amount of stress and recovery I have experienced.

The ultimate goal of this project is to improve my overall fitness and health, while reducing the risk of injury. With a data-driven approach to my training, I believe that I will be able to achieve these goals and reach my full potential as an athlete. As I continue to collect data and refine my model, I look forward to seeing the results and improving my overall fitness and well-being.

## Installation

1. Create an account on Strava and go to https://www.strava.com/settings/api to obtain your API key. Follow the instructions on this [page](https://www.youtube.com/watch?v=sgscChKfGyg) to grant necessary permissions and retrieve the tokens you need.


2. Clone or download this repository.

3. Install the required libraries using the command: pip install -r requirements.txt.

4. Change the following fields in the strava_api.py file to your own Strava API credentials: client_id, client_secret, refresh_token.

5. Run the strava_exploration.py script to analyze your Strava runs.

## Sample Visualizations of Personal Activity Data
<img src="https://user-images.githubusercontent.com/76894056/229618444-fc8c422b-0e5b-4373-a410-9640b5534326.png" alt="Pace over Time" width="500"/>
<img src="https://user-images.githubusercontent.com/76894056/229618498-8ade7bf4-42f7-47f2-8801-825fdb7c695e.png" alt="Cadence vs Distance" width="500"/>
<img src="https://user-images.githubusercontent.com/76894056/229618544-219f5bb3-98d8-45df-97b6-0e03e4b96a2c.png" alt="Suffer Score vs Distance" width="500"/>
<img src="https://github.com/trevorlomba/Strava-Explora/blob/main/visualizations/4.3.23%20Max%20HR%20vs%20Distance.png?raw=true" alt="Max HR vs Distance" width="500"/>
<img src="https://github.com/trevorlomba/Strava-Explora/blob/main/visualizations/4.3.23%20Max%20HR%20vs%20Suffer%20Score.png?raw=true" alt="Max HR vs Suffer Score" width="500"/>
    
## Usage

The strava_exploration.py script retrieves all of your runs for the current year and creates a dataframe with the following columns: name, average_speed, suffer_score, upload_id, type, distance, moving_time, max_speed, total_elevation_gain, start_date_local, average_heartrate, max_heartrate, workout_type, elapsed_time, and average_cadence.

The script then cleans and manipulates the data by converting the date column into a datetime object and breaking it into separate date and time columns, converting distance from meters to miles, calculating pace, and creating additional columns for plotting.

Finally, the script plots the data in multiple graphs.


## Credits

This project was created by Trevor Lomba (tjlomba95@gmail.com) as a personal project. Special thanks to Strava for providing the API and to the creators of the Python libraries used in this project.
