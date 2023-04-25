from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import data_processing
from strava_api import get_strava_data


app = Flask(__name__)
CORS(app, resources={
     r"/api/*": {"origins": ["http://localhost:3000", "https://trevorlomba.github.io"]}})



@app.route('/api/mileage-report')
def mileage_report():
    strava_data = get_strava_data()
    data = jsonify(data_processing.get_mileage_report_data(strava_data))
    return data


@app.route('/api/cadence-report')
def cadence_report():
    strava_data = get_strava_data()
    data = jsonify(data_processing.get_cadence_report_data(strava_data))
    return data


@app.route('/images/<path:image_name>', methods=['GET'])
def get_image(image_name):
    return send_from_directory('visualizations', image_name)


if __name__ == '__main__':
    app.run(debug=True)
