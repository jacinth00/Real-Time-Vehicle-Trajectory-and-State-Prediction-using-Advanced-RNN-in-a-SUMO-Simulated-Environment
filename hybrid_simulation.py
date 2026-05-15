import os
import sys
import time
import numpy as np
import tensorflow as tf

# --- TraCI import and SUMO setup ---
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME' pointing to your SUMO installation.")

import traci

# Choose sumo-gui so the GUI opens
sumoBinary = "sumo-gui"
sumoCmd = [sumoBinary, "-c", "hybrid_network.sumocfg", "--start"]

# --- Load the ML model (your file) ---
MODEL_FILENAME = "Levin_allcars_lstm.h5"
if not os.path.exists(MODEL_FILENAME):
    sys.exit(f"Model file '{MODEL_FILENAME}' not found in the working directory.")

# When loading custom metrics, ensure they are available or set compile=False
model = tf.keras.models.load_model(MODEL_FILENAME, compile=False)

# --- Recreate StandardScaler used in training ---
# (Use the same mean_ and scale_ arrays you used earlier. Adjust if needed.)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.mean_ = np.array([25.7, 0.0, 81.0, 40.78431373, 34.0, 106.0, 0.0, 1600.5, 47.0, 0.0, 0.0, 25.7, 0.0])
scaler.scale_ = np.array([39.7, 1.0, 2.0, 27.45098039, 14.0, 41.0, 1.0, 1189.3125, 73.0, 1.0, 1.0, 39.7, 1.0])

# --- Vehicles to control (two ML vehicles) ---
ml_vehicles = ["ego_0", "ego_1"]

# --- Helper: prepare feature vector for a vehicle ---
def make_features_for_vehicle(current_speed):
    # Build the 13-feature sample used in your earlier script.
    # Adjust values if your model expects different features.
    features = np.array([[current_speed, 0.0, 81.0, 40.78, 34.0, 106.0, 0.0, 1600.5,
                          47.0, 0.0, 0.0, current_speed, 0.0]])
    return features

# --- Start SUMO and enter simulation loop ---
print("Starting SUMO-GUI...")
traci.start(sumoCmd)
step = 0
max_steps = 1000

try:
    while traci.simulation.getMinExpectedNumber() > 0 and step < max_steps:
        traci.simulationStep()

        for vid in ml_vehicles:
            if vid in traci.vehicle.getIDList():
                try:
                    current_speed = traci.vehicle.getSpeed(vid)  # m/s
                    # Build feature vector, scale and shape for LSTM
                    feat = make_features_for_vehicle(current_speed)
                    scaled = scaler.transform(feat)
                    X_pred = scaled.reshape(1, 1, scaled.shape[1])

                    # Predict (assuming model outputs two features where first is next speed)
                    predicted_scaled = model.predict(X_pred, verbose=0)

                    # Workaround: fill full set then inverse_transform
                    full_scaled = np.copy(scaled)
                    # If the model outputs 1 or more values, adapt indexing:
                    # here we assume predicted_scaled[0,0] is next-speed (scaled) and next output maybe other var.
                    # If your model outputs a single value, adjust accordingly.
                    try:
                        full_scaled[0, 0] = predicted_scaled[0, 0]
                        if predicted_scaled.shape[1] > 1:
                            full_scaled[0, 1] = predicted_scaled[0, 1]
                    except Exception:
                        # If prediction shape doesn't match expectation, fallback to using predicted value in first slot
                        full_scaled[0, 0] = float(predicted_scaled.flatten()[0])

                    predicted_features = scaler.inverse_transform(full_scaled)
                    predicted_speed = float(predicted_features[0, 0])

                    # Safety clamp: predicted speed must be between 0 and vehicle max speed
                    vtype = traci.vehicle.getTypeID(vid)
                    try:
                        vmax = traci.vehicletype.getMaxSpeed(vtype)
                    except traci.TraCIException:
                        vmax = 33.33
                    predicted_speed = max(0.0, min(predicted_speed, vmax))

                    # Set speed for next step
                    traci.vehicle.setSpeed(vid, predicted_speed)

                    print(f"[Step {step}] {vid} cur={current_speed:.2f} -> pred={predicted_speed:.2f}")
                except traci.TraCIException as e:
                    print(f"Traci exception for {vid}: {e}")

        step += 1

except KeyboardInterrupt:
    print("Interrupted by user.")
finally:
    traci.close()
    print("Simulation closed.")
