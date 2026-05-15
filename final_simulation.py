import os
import sys
import traci
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# --- SUMO Configuration ---
# Make sure the SUMO_HOME environment variable is set
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

sumoBinary = "sumo-gui"  # Use "sumo" for a non-GUI version
sumoCmd = [sumoBinary, "-c", "final_network.sumocfg"]

# --- Load Model and Recreate Scaler ---
model = load_model(
    'Levin_allcars_lstm.h5',
    custom_objects={'mse': tf.keras.metrics.MeanSquaredError}
)

# Manually recreate the StandardScaler with your values
scaler = StandardScaler()
scaler.mean_ = np.array([25.7, 0.0, 81.0, 40.78431373, 34.0, 106.0, 0.0, 1600.5, 47.0, 0.0, 0.0, 25.7, 0.0])
scaler.scale_ = np.array([39.7, 1.0, 2.0, 27.45098039, 14.0, 41.0, 1.0, 1189.3125, 73.0, 1.0, 1.0, 39.7, 1.0])

# --- Simulation Loop ---
traci.start(sumoCmd)
step = 0
vehicle_id = "ego_0"

while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()

    # Check if our vehicle is still in the simulation
    if vehicle_id in traci.vehicle.getIDList():
        current_speed = traci.vehicle.getSpeed(vehicle_id)
        pos = traci.vehicle.getPosition(vehicle_id)
        try:
            longitude, latitude = traci.simulation.convertGeo(pos[0], pos[1])
        except traci.TraCIException:
            longitude, latitude = 0.0, 0.0

        # Create the full 13-feature vector for the model
        features = np.array([[
            current_speed, 0.0, 81.0, 40.78, 34.0, 106.0, 0.0, 1600.5,
            47.0, 0.0, 0.0, current_speed, 0.0
        ]])

        # Scale the features and reshape for the LSTM model
        scaled_features = scaler.transform(features)
        X_pred = scaled_features.reshape(1, 1, scaled_features.shape[1])
        
        # Make a prediction
        predicted_scaled = model.predict(X_pred, verbose=0)

        # WORKAROUND: Create a full set of 13 features for the scaler
        full_prediction_set = np.copy(scaled_features)
        
        # Update the first two features with the model's prediction
        full_prediction_set[0, 0] = predicted_scaled[0, 0] 
        full_prediction_set[0, 1] = predicted_scaled[0, 1] 
        
        # Inverse_transform the complete 13-feature set
        predicted_features = scaler.inverse_transform(full_prediction_set)
        
        # Extract the unscaled speed from the full set
        predicted_speed = predicted_features[0, 0]

        # Set the vehicle's speed in SUMO for the next step
        traci.vehicle.setSpeed(vehicle_id, predicted_speed)

        print(f"Step {step}: Current Speed={current_speed:.2f}, Predicted Next Speed={predicted_speed:.2f}")

    step += 1

traci.close()
print("Simulation finished.")