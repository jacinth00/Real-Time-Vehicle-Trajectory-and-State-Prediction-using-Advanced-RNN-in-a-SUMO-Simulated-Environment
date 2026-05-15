# robust_hybrid_simulation.py
import os
import sys
import time
import csv
import numpy as np
import joblib

# TraCI / SUMO setup
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME' pointing to your SUMO installation.")

import traci

# ML imports (deferred until after verifying files)
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# --- Configuration (edit if your filenames differ) ---
MODEL_CANDIDATES = [
    "Levin_allcars_lstm_trajectory.h5",
    "Levin_allcars_lstm.h5",
    "Levin_allcars.h5",
    "Levin_allcars_lstm.pkl"  # sometimes saved as different name
]
SCALER_CANDIDATES = [
    "Levin_allcars_scaler_trajectory.pkl",
    "Levin_allcars_scaler.pkl",
    "Levin_allcars_scaler.pickle"
]
SUMOCFG = "hybrid_network.sumocfg"
LOG_CSV = "simulation_results.csv"
ml_vehicles = ["ego_0", "ego_1"]   # vehicles to control (must exist in routes)

# --- Helper: find model & scaler files ---
def find_existing(path_list):
    for p in path_list:
        if os.path.exists(p):
            return p
    return None

MODEL_FILENAME = find_existing(MODEL_CANDIDATES)
SCALER_FILENAME = find_existing(SCALER_CANDIDATES)

if MODEL_FILENAME is None:
    sys.exit("Model file not found. Place your .h5 model in the working directory and try again. "
             f"Tried: {MODEL_CANDIDATES}")

print(f"Loading model from: {MODEL_FILENAME}")
model = tf.keras.models.load_model(MODEL_FILENAME, compile=False)
# Print model summary to help debug shape expectations
print("Model summary:")
model.summary()

# --- Load scaler if present; else try to create fallback StandardScaler with attributes (if you know them) ---
scaler = None
if SCALER_FILENAME:
    try:
        scaler = joblib.load(SCALER_FILENAME)
        print(f"Loaded scaler from: {SCALER_FILENAME}")
    except Exception as e:
        print(f"Failed to load scaler {SCALER_FILENAME}: {e}")
else:
    print("No scaler file found in candidates:", SCALER_CANDIDATES)

# If no scaler was loaded, you can optionally set a fallback scaler with mean_/scale_.
# WARNING: Only do this if you know the exact training scaler values. Otherwise predictions will be wrong.
if scaler is None:
    print("No scaler loaded. Creating fallback StandardScaler (UNTRAINED).")
    scaler = StandardScaler()
    # DO NOT set mean_/scale_ unless you are certain. Example (commented):
    # scaler.mean_ = np.array([...])
    # scaler.scale_ = np.array([...])
    # scaler.n_features_in_ = <number>
    # If you don't set these, scaler.transform will raise an error; we'll check below.

# --- Feature builder ---
# IMPORTANT: This must match exactly the feature order used when training your model.
# Update input_cols and make_features_for_vehicle if your training used a different order.
input_cols = ['gps_speed', 'battery', 'cTemp', 'eLoad',
              'iat', 'imap', 'maf', 'rpm', 'speed', 'tAdv', 'tPos']
# If scaler was fitted on input+target concatenation, change combined_len accordingly
# Many training flows used input + target concatenated; set combined_len appropriately:
combined_len = None
if hasattr(scaler, 'n_features_in_'):
    combined_len = scaler.n_features_in_
    print("Scaler expects", combined_len, "features (n_features_in_)")
else:
    # If scaler was fitted on input+target columns and saved, set the expected length manually:
    # Example: input_len = len(input_cols); target_len = 2 -> combined_len = input_len + target_len
    combined_len = len(input_cols) + 2  # assume 2 targets (speed, rpm)
    print("Assuming scaler expects", combined_len, "features (input + targets).")

def make_features_for_vehicle(current_speed):
    """
    Build the feature vector in the same order used by training.
    Values other than current_speed are placeholders and should be changed
    to read real sensor values if available.
    """
    # placeholder values; if you recorded other signals in training, substitute them here
    feat = [
        current_speed,   # gps_speed
        0.0,             # battery
        66.0,            # cTemp
        0.0,             # eLoad
        40.0,            # iat
        97.0,            # imap
        0.0,             # maf
        1000.0,          # rpm (placeholder)
        current_speed,   # speed (duplicate sometimes used)
        0.0,             # tAdv (heading)
        0.0              # tPos (position index)
    ]
    return np.array([feat], dtype=float)

# --- SUMO start command ---
if not os.path.exists(SUMOCFG):
    print(f"Warning: SUMO config '{SUMOCFG}' not found in working directory.")
sumoBinary = "sumo-gui"
sumoCmd = [sumoBinary, "-c", SUMOCFG, "--start"]

# --- CSV logging setup ---
csv_fields = ["step", "vehicle_id", "pred_speed", "pred_rpm", "applied_speed"]
csv_file = open(LOG_CSV, mode="w", newline="")
csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
csv_writer.writeheader()

# --- Start SUMO and control loop ---
print("Starting SUMO-GUI...")
traci.start(sumoCmd)

# give SUMO one step to initialize vehicles
traci.simulationStep()
current_veh_list = traci.vehicle.getIDList()
print("Vehicles currently in simulation at start:", current_veh_list)

# Wait until ML vehicles appear (timeout if they don't)
timeout_steps = 10
waited = 0
while not any(v in current_veh_list for v in ml_vehicles) and waited < timeout_steps:
    traci.simulationStep()
    waited += 1
    current_veh_list = traci.vehicle.getIDList()
    print(f"Waiting for ML vehicles... step {waited}, present: {current_veh_list}")

if not any(v in current_veh_list for v in ml_vehicles):
    print("⚠️ ML vehicles not present after waiting. Check your routes file (.rou.xml) to ensure ego_0 and ego_1 depart at or near t=0.")
    # do not immediately exit; continue but nothing will be controlled

step = 0
max_steps = 1000

try:
    while traci.simulation.getMinExpectedNumber() > 0 and step < max_steps:
        traci.simulationStep()

        veh_ids = traci.vehicle.getIDList()
        for vid in ml_vehicles:
            if vid not in veh_ids:
                continue  # skip if not present yet

            try:
                # read current speed (m/s)
                cur_speed = traci.vehicle.getSpeed(vid)

                # build features & scale
                feat = make_features_for_vehicle(cur_speed)   # shape (1, input_len)
                # We assume scaler was fitted on input + target combined. We must provide same width.
                # Create combined array by concatenating feat with placeholder zeros for targets
                if combined_len is not None and feat.shape[1] < combined_len:
                    pad = np.zeros((1, combined_len - feat.shape[1]))
                    combined = np.concatenate([feat, pad], axis=1)
                else:
                    combined = feat

                # transform and reshape for LSTM (1, timesteps, features)
                # Note: many models were trained with timesteps=1; adapt if you used sequences >1
                try:
                    scaled = scaler.transform(combined)
                except Exception as e:
                    print("Scaler.transform failed:", e)
                    print("Scaler details:", type(scaler), getattr(scaler, 'n_features_in_', None))
                    raise

                X_pred = scaled.reshape(1, 1, scaled.shape[1])

                # Predict
                pred = model.predict(X_pred, verbose=0)  # shape (1, n_outputs) or (1,1,n_outputs)

                # flatten and detect shape
                pred_arr = np.array(pred).reshape(-1)
                print(f"DEBUG: raw predicted array for {vid}:", pred_arr)

                # Map predicted outputs to variables
                # If model predicts only speed, RPM absent -> set rpm = NaN
                pred_speed_scaled = pred_arr[0] if pred_arr.size >= 1 else None
                pred_rpm_scaled = pred_arr[1] if pred_arr.size >= 2 else None

                # Insert predicted values back into scaled combined to inverse_transform
                inv_scaled = np.copy(scaled)
                if pred_speed_scaled is not None:
                    inv_scaled[0, 0] = pred_speed_scaled
                if pred_rpm_scaled is not None and inv_scaled.shape[1] > 1:
                    inv_scaled[0, 1] = pred_rpm_scaled

                # inverse transform to original units (guard with try)
                try:
                    inv = scaler.inverse_transform(inv_scaled)
                except Exception as e:
                    print("scaler.inverse_transform failed:", e)
                    inv = np.zeros_like(inv_scaled)  # fallback to zeros

                applied_speed = float(inv[0, 0]) if inv.shape[1] > 0 else float(cur_speed)
                applied_rpm = float(inv[0, 1]) if inv.shape[1] > 1 else float('nan')

                # clamp applied_speed
                try:
                    vtype = traci.vehicle.getTypeID(vid)
                    vmax = traci.vehicletype.getMaxSpeed(vtype)
                except Exception:
                    vmax = 33.33
                applied_speed = max(0.0, min(applied_speed, vmax))

                # set vehicle speed
                traci.vehicle.setSpeed(vid, applied_speed)

                # print and log
                print(f"[Step {step:04d}] {vid} | pred_speed={applied_speed:.3f} m/s | pred_rpm={applied_rpm:.2f}")
                csv_writer.writerow({
                    "step": step,
                    "vehicle_id": vid,
                    "pred_speed": applied_speed,
                    "pred_rpm": applied_rpm,
                    "applied_speed": applied_speed
                })

            except Exception as e:
                print(f"⚠️ Error controlling vehicle {vid} at step {step}: {e}")

        step += 1

except KeyboardInterrupt:
    print("Simulation interrupted by user.")
finally:
    try:
        csv_file.close()
    except:
        pass
    traci.close()
    print("Simulation closed. Results logged to", LOG_CSV)
