from flask import Flask, request
import os
import csv
# Sample file to record the data from the mobile app


app = Flask(__name__)
DATA_DIR = "sensor_csv_logs"
os.makedirs(DATA_DIR, exist_ok=True)

# Only save these sensors
ALLOWED_SENSORS = {"accelerometer", "gyroscope"}

def write_sensor_data(sensor_name, timestamp, accuracy, values):
    filename = os.path.join(DATA_DIR, f"{sensor_name}.csv")

    row = {'time': timestamp, 'accuracy': accuracy}
    row.update(values)

    fieldnames = ['time'] + list(values.keys()) + ['accuracy']
    file_exists = os.path.exists(filename)

    with open(filename, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

@app.route('/sensor', methods=['POST'])
def handle_sensor_post():
    data = request.get_json(force=True)

    if "payload" not in data:
        return "Invalid data: no payload", 400

    for entry in data["payload"]:
        sensor_name = entry.get("name", "unknown").lower()

        if sensor_name not in ALLOWED_SENSORS:
            continue  # Skip if not accelerometer or gyroscope

        timestamp = entry.get("time", "")
        accuracy = entry.get("accuracy", "")
        values = entry.get("values", {})

        write_sensor_data(sensor_name, timestamp, accuracy, values)

    return "OK", 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
