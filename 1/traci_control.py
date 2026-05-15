import traci
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Start SUMO with TraCI
sumoCmd = ["C:\\Program Files (x86)\\Eclipse\\Sumo\\bin\\sumo-gui.exe", "-c", "mySimulation.sumocfg"]
traci.start(sumoCmd)

# Load ML model and scaler
model = load_model("NGSIM_lstm.h5")
scaler = joblib.load("NGSIM_scaler.pkl")   # adjust file names if needed

# Simulation loop
for step in range(1000):
    traci.simulationStep()
    
    # For ML car
    ml_id = "ml_1"
    if ml_id in traci.vehicle.getIDList():
        # Example: get position, speed, etc.
        position = traci.vehicle.getPosition(ml_id)
        speed = traci.vehicle.getSpeed(ml_id)
        # Prepare features for prediction
        features = np.array([[speed]]) # add other features if needed

        # Scale features
        features_scaled = scaler.transform(features)
        # Predict next speed (example, adapt for your output)
        pred_speed = model.predict(features_scaled)
        
        # Set speed for ML car
        traci.vehicle.setSpeed(ml_id, float(pred_speed[0][0]))
    
    # For Default car: no manual control (SUMO controls as per vType)
    
    # (Optional) Log/Print vehicle states, speeds, positions
    
traci.close()
