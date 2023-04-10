import data_processing

activities = data_processing.get_activities()
print(activities)

mileage_report = data_processing.get_mileage_report_data()
print(mileage_report)

cadence_report = data_processing.get_cadence_report_data()
print(cadence_report)
