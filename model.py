from tensorflow.keras.models import load_model
import pickle

# Load the trained model
model = load_model('Levin_allcars_lstm.h5')

# Load the scaler
with open('Levin_allcars_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)