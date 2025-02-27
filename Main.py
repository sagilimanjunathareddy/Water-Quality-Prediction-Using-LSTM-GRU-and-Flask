import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense


print(" Loading dataset...")
if not os.path.exists("water_potability.csv"):
    raise FileNotFoundError("🚨 Error: 'water_potability.csv' not found! Please add the dataset.")

df = pd.read_csv('water_potability.csv').dropna()  
X = df.drop('Potability', axis=1)
Y = df['Potability'].astype(int)
print("✅ Dataset loaded successfully!")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print("📊 Scaling data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Scaler saved as 'scaler.pkl'!")

# Reshape data for LSTM/GRU
X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

# Train LSTM Model
print("📈 Training LSTM model...")
model_lstm = Sequential([
    LSTM(50, input_shape=(1, X_train_scaled.shape[1])),
    Dense(1, activation='sigmoid')
])
model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_lstm.fit(X_train_reshaped, Y_train, epochs=50, batch_size=16, verbose=1)

#  Save LSTM model
model_lstm.save('lstm_model.h5')
print("LSTM model saved as 'lstm_model.h5'!")

#  Train GRU Model
print("📈 Training GRU model...")
model_gru = Sequential([
    GRU(50, input_shape=(1, X_train_scaled.shape[1])),
    Dense(1, activation='sigmoid')
])
model_gru.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_gru.fit(X_train_reshaped, Y_train, epochs=50, batch_size=16, verbose=1)

#  Save GRU model
model_gru.save('gru_model.h5')
print(" GRU model saved as 'gru_model.h5'!")

print(" Training complete! All models and scaler have been saved.")
