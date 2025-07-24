from flask import Flask, render_template, request, redirect, url_for, jsonify, session, send_from_directory
import os
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd
from datetime import datetime
import numpy as np

app = Flask(__name__)
app.secret_key = 'supersecretkey' 

# Different set of poses used/classes used, bend_forward_down, twisting, quick_stand_sit, asymmetric_movement are the risky classes
# we can add more classes as needed
POSES = [
    'bend_forward_down',
    'twisting',
    'quick_stand_sit',
    'normal_sit',
    'normal_stand',
    'normal_walk',
    'asymmetric_movement',
    'normalspeedsit'
] 

# Sensitivity settings for each class
SENSITIVITY = {pose: 0.5 for pose in POSES}

LABELED_DATA_DIR = 'labeled_data'
os.makedirs(LABELED_DATA_DIR, exist_ok=True)

MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Global state for recording
RECORDING = False
CURRENT_POSE = None

LIVE_PREDICT = False
LIVE_MODEL_PATH = None
LAST_PREDICTION = None

# Live data buffers and stats for dsplaying it in the live page
LIVE_ACC = {'z': [], 'y': [], 'x': [], 't': []}
LIVE_GYRO = {'z': [], 'y': [], 'x': [], 't': []}
LIVE_POSTURE_COUNTS = {}
LIVE_HISTORY = []
LIVE_LAST_POSTURE = None
LIVE_MAX_POINTS = 100

@app.route('/')
def landing():
    return render_template('landing.html') #Main Page

@app.route('/home')
def home():
    return render_template('home.html', poses=POSES) # Data Recording Page
# Reset data to remove all the current data and train a new model
@app.route('/reset_data', methods=['POST'])
def reset_data():
    for fname in os.listdir(LABELED_DATA_DIR):
        fpath = os.path.join(LABELED_DATA_DIR, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)
    return redirect(url_for('home'))

@app.route('/train', methods=['GET', 'POST'])
def train():
    trained = False
    accuracy = None
    error = None
    if request.method == 'POST':
        model_name = request.form.get('model_name', '').strip()
        if not model_name:
            error = 'Model name is required.'
            return render_template('train.html', trained=False, error=error)
        acc_path = os.path.join(LABELED_DATA_DIR, 'accelerometer.csv')
        gyro_path = os.path.join(LABELED_DATA_DIR, 'gyroscope.csv')
        if not os.path.exists(acc_path) or not os.path.exists(gyro_path):
            return render_template('train.html', trained=False, error='No data to train on.')
        acc_df = pd.read_csv(acc_path)
        gyro_df = pd.read_csv(gyro_path)
        merged = pd.merge(acc_df, gyro_df, left_index=True, right_index=True, suffixes=('_acc', '_gyro')) 
        merged = merged[merged['pose_acc'] == merged['pose_gyro']] # Combine the both accelerometer and gyroscope data
        merged['pose'] = merged['pose_acc']
        X = merged[['z_acc', 'y_acc', 'x_acc', 'z_gyro', 'y_gyro', 'x_gyro']] #Input Features
        y = merged['pose'] # Output Class
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42) #RFC is used for classification
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        model_path = os.path.join(MODEL_DIR, f'{model_name}.joblib')
        joblib.dump(clf, model_path)
        trained = True
        return render_template('train.html', trained=trained, accuracy=accuracy, model_name=model_name)
    return render_template('train.html', trained=trained) # Training Page

@app.route('/live', methods=['GET', 'POST']) # Live Detection Page
def live():
    global LIVE_PREDICT, LIVE_MODEL_PATH, LAST_PREDICTION, LIVE_ACC, LIVE_GYRO, LIVE_POSTURE_COUNTS, LIVE_HISTORY, LIVE_LAST_POSTURE
    models = [f for f in os.listdir(MODEL_DIR) if f.endswith('.joblib')] #Load the model
    prediction = LAST_PREDICTION
    selected_model = None
    if request.method == 'POST':
        selected_model = request.form.get('model', None)
        if not selected_model or not os.path.exists(os.path.join(MODEL_DIR, selected_model)):
            return render_template('live.html', prediction=prediction, live=False, models=models, error='Model not found.', poses=POSES)
        model_path = os.path.join(MODEL_DIR, selected_model)
        LIVE_PREDICT = True
        LIVE_MODEL_PATH = model_path
        # Reset live data buffers and stats
        LIVE_ACC = {'z': [], 'y': [], 'x': [], 't': []}
        LIVE_GYRO = {'z': [], 'y': [], 'x': [], 't': []}
        LIVE_POSTURE_COUNTS = {}
        LIVE_HISTORY = []
        LIVE_LAST_POSTURE = None
        return render_template('live.html', prediction=prediction, live=True, models=models, selected_model=selected_model, poses=POSES)
    return render_template('live.html', prediction=prediction, live=False, models=models, selected_model=selected_model, poses=POSES)

@app.route('/set_sensitivity', methods=['POST']) #Adjust sensitivity of the classes
def set_sensitivity():
    global SENSITIVITY
    data = request.get_json(force=True)
    for pose in POSES:
        if pose in data:
            try:
                SENSITIVITY[pose] = float(data[pose])
            except Exception:
                pass
    return jsonify({'ok': True, 'sensitivity': SENSITIVITY})

@app.route('/live_data') #Live data for the live page
def live_data():
    global LIVE_ACC, LIVE_GYRO, LIVE_POSTURE_COUNTS, LIVE_HISTORY, LIVE_LAST_POSTURE, SENSITIVITY
    return jsonify({
        'acc': LIVE_ACC,
        'gyro': LIVE_GYRO,
        'pie': LIVE_POSTURE_COUNTS,
        'history': LIVE_HISTORY,
        'current_posture': LIVE_LAST_POSTURE,
        'sensitivity': SENSITIVITY
    })

@app.route('/predict_sensor', methods=['POST']) #For Fetching data from HTTP POST request from mobile app
def predict_sensor():
    global LIVE_PREDICT, LIVE_MODEL_PATH, LAST_PREDICTION, LIVE_ACC, LIVE_GYRO, LIVE_POSTURE_COUNTS, LIVE_HISTORY, LIVE_LAST_POSTURE
    if not LIVE_PREDICT or not LIVE_MODEL_PATH:
        return jsonify({'prediction': None, 'error': 'Live prediction not active'}), 200
    data = request.get_json(force=True)
    if "payload" not in data:
        return jsonify({'prediction': None, 'error': 'No payload'}), 400
    model = joblib.load(LIVE_MODEL_PATH)
    acc = None
    gyro = None
    timestamp = datetime.now().strftime('%H:%M:%S')
    for entry in data["payload"]:
        sensor_name = entry.get("name", "unknown").lower()
        values = entry.get("values", {})
        if sensor_name == "accelerometer":
            acc = values
        elif sensor_name == "gyroscope":
            gyro = values
    if acc and gyro:
        # Use DataFrame with correct feature names
        X = pd.DataFrame([[acc.get('z', 0), acc.get('y', 0), acc.get('x', 0), gyro.get('z', 0), gyro.get('y', 0), gyro.get('x', 0)]],
                         columns=['z_acc','y_acc','x_acc','z_gyro','y_gyro','x_gyro'])
        pred = model.predict(X)[0]
        LAST_PREDICTION = pred
        # Update live data buffers
        now = datetime.now()
        tstr = now.strftime('%H:%M:%S')
        for axis in ['z','y','x']:
            LIVE_ACC[axis].append(acc.get(axis, 0))
            LIVE_GYRO[axis].append(gyro.get(axis, 0))
        LIVE_ACC['t'].append(tstr)
        LIVE_GYRO['t'].append(tstr)
        # Keep only last N points
        for d in [LIVE_ACC, LIVE_GYRO]:
            for k in d:
                if len(d[k]) > LIVE_MAX_POINTS:
                    d[k] = d[k][-LIVE_MAX_POINTS:]
        # Update posture counts
        if pred not in LIVE_POSTURE_COUNTS:
            LIVE_POSTURE_COUNTS[pred] = 0
        LIVE_POSTURE_COUNTS[pred] += 1
        # Update history if posture changed
        if pred != LIVE_LAST_POSTURE:
            LIVE_LAST_POSTURE = pred
            dt = now.strftime('%Y-%m-%d')
            tm = now.strftime('%H:%M:%S')
            LIVE_HISTORY.append({'date': dt, 'time': tm, 'pose': pred})
        return jsonify({'prediction': pred})
    else:
        return jsonify({'prediction': None, 'error': 'Need both accelerometer and gyroscope data'}), 200

@app.route('/get_prediction', methods=['GET'])
def get_prediction():
    global LAST_PREDICTION
    return jsonify({'prediction': LAST_PREDICTION})

@app.route('/record', methods=['POST']) #Record the data from the mobile app
def record():
    global RECORDING, CURRENT_POSE
    pose = request.form.get('pose')
    if pose not in POSES:
        return redirect(url_for('home'))
    CURRENT_POSE = pose
    RECORDING = True
    return render_template('home.html', poses=POSES, recording=True, selected_pose=pose)

@app.route('/stop_recording', methods=['POST']) #Stop recording the data from the mobile app
def stop_recording():
    global RECORDING, CURRENT_POSE
    RECORDING = False
    CURRENT_POSE = None
    return render_template('home.html', poses=POSES, recording=False, selected_pose=None)

@app.route('/sensor', methods=['POST']) #Handle the sensor data from the mobile app
def handle_sensor_post():
    global RECORDING, CURRENT_POSE
    if not RECORDING or not CURRENT_POSE:
        return 'Not recording', 200
    data = request.get_json(force=True)
    if "payload" not in data:
        return "Invalid data: no payload", 400
    pose = CURRENT_POSE
    for entry in data["payload"]:
        sensor_name = entry.get("name", "unknown").lower()
        if sensor_name not in {"accelerometer", "gyroscope"}:
            continue
        values = entry.get("values", {})
        row = {
            'z': values.get('z', ''),
            'y': values.get('y', ''),
            'x': values.get('x', ''),
            'pose': pose
        }
        filename = os.path.join(LABELED_DATA_DIR, f"{sensor_name}.csv")
        file_exists = os.path.exists(filename)
        with open(filename, mode='a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['z', 'y', 'x', 'pose'])
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    return "OK", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 