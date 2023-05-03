import data_processing
import strava_api

# activities = data_processing.get_activities()
# print(activities)
my_dataset = strava_api.get_strava_data()

mileage_report = data_processing.get_mileage_report_data(my_dataset)
print(mileage_report)
print('mileage_report')

cadence_report = data_processing.get_cadence_report_data(my_dataset)
print(cadence_report)
