from flask import Flask, jsonify
import data_processing

app = Flask(__name__)


@app.route('/api/mileage-report')
def mileage_report():
    data = data_processing.get_mileage_report_data()
    return jsonify(data)

@app.route('/api/cadence-report')
def cadence_report():
    data = data_processing.get_cadence_report_data()
    return jsonify(data)


if __name__ == '__main__':
    app.run()
