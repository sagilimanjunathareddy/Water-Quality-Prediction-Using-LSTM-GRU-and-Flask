import os
from flask import Flask, render_template, request
import numpy as np
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__, template_folder='templates', static_folder='static')

# Ensure the model files exist before loading
if not os.path.exists("lstm_model.h5") or not os.path.exists("gru_model.h5"):
    raise FileNotFoundError("Model files not found! Train the models first by running Main.py.")

# Load models and scaler
model_lstm = load_model('lstm_model.h5')
model_gru = load_model('gru_model.h5')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('Main.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        values = np.array([
            float(request.form.get("ph")),
            float(request.form.get("hardness")),
            float(request.form.get("solids")),
            float(request.form.get("chloramines")),
            float(request.form.get("sulfate")),
            float(request.form.get("conductivity")),
            float(request.form.get("organic_carbon")),
            float(request.form.get("trihalomethanes")),
            float(request.form.get("turbidity"))
        ]).reshape(1, -1)

        values_scaled = scaler.transform(values)
        values_reshaped = values_scaled.reshape(1, 1, values_scaled.shape[1])

        lstm_prediction = model_lstm.predict(values_reshaped)[0][0]
        gru_prediction = model_gru.predict(values_reshaped)[0][0]

        lstm_result = "Potable Water" if lstm_prediction >= 0.5 else "Non-Potable Water"
        gru_result = "Potable Water" if gru_prediction >= 0.5 else "Non-Potable Water"

        return render_template('Result.html', lstm_result=lstm_result, gru_result=gru_result)

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
