# PregPose Pal
![WhatsApp Image 2025-07-24 at 23 30 09_a0ee4a6e](https://github.com/user-attachments/assets/06b6c220-b9c4-40a2-a8e5-3bad8c45562b)


## Overview
PregPose Pal is a web-based platform designed to help pregnant women monitor and improve their posture and movement safety using only their mobile phone sensors. The system enables users to record their own movement data, train custom machine learning models, and receive real-time feedback and visualizations to help prevent falls and discomfort during pregnancy.

## Motivation
Falls and unsafe postures are a significant risk during pregnancy, potentially leading to injury for both mother and baby. Many existing solutions require expensive wearables or are not personalized. PregPose Pal leverages the sensors already present in smartphones to provide a low-cost, accessible, and customizable solution for posture and movement monitoring.

## Technologies Used
- **Flask**: Python web framework for backend API and UI rendering
- **scikit-learn**: Machine learning library for model training and prediction (Random Forest)
- **pandas, numpy**: Data processing and feature engineering
- **Chart.js**: Frontend JavaScript library for animated, interactive charts
- **HTML/CSS/JS**: Responsive, modern user interface
- **Sensor Logger App**: Third-party mobile app for streaming accelerometer and gyroscope data to the server

<img src="https://github.com/user-attachments/assets/bdc1ce51-1ce3-498a-8da6-d670cc572fe5" width="300">
<img src="https://github.com/user-attachments/assets/40208c70-27b7-4a80-9480-daa1ec149275" width="300">



## Project Structure
- `app.py`: Main Flask application (all backend logic)
- `templates/`: HTML templates for all pages (Jinja2)
- `static/`: Static assets (icons, pose images, CSS, JS)
- `labeled_data/`: Directory for user-recorded sensor data (CSV)
- `models/`: Directory for trained machine learning models
- `pregnancy_pose_analysis.ipynb`: Jupyter notebook for data analysis and feature importance
- `record_acl_gyro.py`: Sample code for testing data transfer from Sensor Logger app

## Setup
1. **Clone the repository**
   ```
   git clone <repo-url>
   cd <repo-folder>
   ```
2. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```
3. **Run the Flask app**
   ```
   python app.py
   ```
   - The app will be available at `http://localhost:5000` (or your network IP for mobile use).
   - If you encounter port issues, ensure port 5000 is open and not in use.
4. **Access from your mobile device**
   - Connect your phone to the same Wi-Fi network as your computer.
   - Use your computer's local IP address (e.g., `http://192.168.1.10:5000`).

## Usage Guide
### 1. **Recording Data**
![WhatsApp Image 2025-07-24 at 23 29 29_c43202d4](https://github.com/user-attachments/assets/7b01f5c6-a57e-416e-afbb-de4973d69d3c)

- Go to the **Record** page.
- Follow the on-screen instructions to set up the Sensor Logger app:
  - Download the app on your mobile device.
  - Enable Gyroscope and Accelerometer.
  - Go to Settings > Data Streaming > HTTP Push.
  - Enter your laptop's IP and `/sensor` as the Push URL (e.g., `http://192.168.1.10:5000/sensor`).
- Select a pose, start recording, and perform the movement for at least 10 seconds.
- Stop recording before switching to another pose.
- Repeat for all poses you want to train.


### Sensor Logger Config
![WhatsApp Image 2025-07-24 at 23 17 44_7133b05b](https://github.com/user-attachments/assets/6a894ba3-1f92-4215-a271-feaf537656e1)

### 2. **Training a Model**
![WhatsApp Image 2025-07-24 at 23 28 44_17a83485](https://github.com/user-attachments/assets/2cb7f4f4-fe3d-45ac-b56b-11e31f114c8f)

- Go to the **Train** page.
- Enter a model name and click "Train Model".
- The system will use your recorded data to train a Random Forest classifier.
- Training accuracy and feature importance are displayed after training.
- Each model is saved and can be selected later for live prediction.

### 3. **Live Prediction**
![WhatsApp Image 2025-07-24 at 23 26 05_58bbd25c](https://github.com/user-attachments/assets/4d992a31-7de6-4d01-9313-205ed2f5b633)

- Go to the **Live Detection** page.
- Change the Sensor Logger app's Push URL to `/predict_sensor` (e.g., `http://192.168.1.10:5000/predict_sensor`).
- Select your trained model and start live prediction.
- The app will display:
  - The current detected posture
  - Live graphs of accelerometer and gyroscope data
  - A pie chart of detected postures
  - A timeline of posture changes (downloadable as CSV)
  - Visual and vibration feedback for risky postures
- You can adjust sensitivity for each posture class in real time.

### Sensor Logger Config
![WhatsApp Image 2025-07-24 at 22 13 02_1e8995d4](https://github.com/user-attachments/assets/35ee09f3-f988-4d97-a453-f697b79b6e1d)


## Pages Explained
- **Landing Page**: Project overview, features, and quick start.
- **Record**: Data collection with pose selection, images, and instructions.
- **Train**: Model training, accuracy display, and model management.
- **Live Detection**: Real-time prediction, feedback, visualization, and history export.

## Model Details
- **Algorithm**: Random Forest Classifier
- **Why Random Forest?**
  - Fast and efficient for medium-sized datasets
  - Robust to noise and outliers (common in sensor data)
  - Easy to interpret (feature importance)
  - Requires little hyperparameter tuning
- **Inputs**: Windowed features from accelerometer and gyroscope (x, y, z axes)
- **Outputs**: Predicted posture class (e.g., bend_forward_down, twisting, quick_stand_sit, normal_sit, normal_stand, normal_walk, asymmetric_movement, normalspeedsit)

## Data Flow
1. **Sensor Logger app** streams data to Flask server (`/sensor` or `/predict_sensor` endpoint).
2. **Flask app** saves data to CSV during recording, or runs live prediction using the selected model.
3. **Frontend** polls the backend for live data, updates charts, and provides feedback.

## Feature Engineering
- Features are extracted from each window of sensor data (e.g., mean, std, min, max for each axis).
- Feature importance (from analysis):

| Feature   | Importance |
|-----------|------------|
| x_gyro    | 0.207      |
| y_gyro    | 0.186      |
| y_accel   | 0.181      |
| x_accel   | 0.162      |
| z_accel   | 0.135      |
| z_gyro    | 0.129      |

## Customization & Extension
- You can add new poses by updating the POSES list in `app.py` and adding images to `static/`.
- To use a different model (e.g., XGBoost, 1D CNN), update the training and prediction logic in `app.py`.
- For more advanced features (user profiles, cloud storage, etc.), extend the Flask app and frontend as needed.
- Implement Seperate Mobile App to record and process all data

## Data and Analysis
- Personal sample data is available in the `labeled_data/` directory.
- Data analysis and feature importance can be found in `pregnancy_pose_analysis.ipynb`.
- The file `record_acl_gyro.py` contains sample code to check data transfer from the Sensor Logger app to Flask.

## Troubleshooting
- **No data recorded?** Check that your phone and computer are on the same network, and the Push URL is correct.
- **Live prediction not working?** Ensure the correct model is selected and the Push URL is set to `/predict_sensor`.
- **Accuracy too low?** Record more data for each pose, and ensure you perform each movement clearly and consistently.


