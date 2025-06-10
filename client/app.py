# client/app.py
from flask import Flask, render_template, request
import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import minmax_scale
from tensorflow.keras.models import Sequential

app = Flask(__name__, template_folder='templates')
SERVER_URL = 'http://192.168.29.80:5000'  

df = pd.read_csv('SaYoPillow.csv')  
X = df.drop('sl', axis=1)
y = df['sl']

X = minmax_scale(X)

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_data = np.array([
            float(request.form['sr']),
            float(request.form['rr']),
            float(request.form['t']),
            float(request.form['lm']),
            float(request.form['bo']),
            float(request.form['rem']),
            float(request.form['sr1']),
            float(request.form['hr'])
        ]).reshape(1, -1)

        prediction = make_prediction(input_data)
        return render_template('result.html', prediction=prediction)
    return render_template('index.html')

def make_prediction(input_data):
    # Step 1: Get weights from server
    res = requests.get(f"{SERVER_URL}/get_weights")
    weights_dict = res.json()
    weights = [np.array(weights_dict[f"w{i}"]) for i in range(len(weights_dict))]

    # Step 2: Create model
    model = Sequential([
        tf.keras.layers.Dense(64, input_shape=(8,), activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    model.set_weights(weights)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Step 3: Local training with dummy label
    history = model.fit(X, y, epochs=10, batch_size=4, verbose=0)
    local_accuracy = history.history['accuracy'][0] 

    # Step 4: Send updated weights to server
    requests.post(f"{SERVER_URL}/send_update", json={
        'weights': [w.tolist() for w in model.get_weights()],
        'local_accuracy': float(local_accuracy)
    })
    # Step 5: Make prediction
    pred = model.predict(input_data)
    return pred.argmax(axis=-1)[0]

    # Step 5: Make prediction
    return model.predict(input_data).argmax(axis=-1)[0]

def get_stress_description(level):
    descriptions = {
        0: "Very Low Stress – You are calm and relaxed. Keep up your healthy routine!",
        1: "Low Stress – Mild stress from daily tasks. Try mindfulness or breathing exercises.",
        2: "Moderate Stress – Noticeable stress from workload or health. Take breaks and reflect.",
        3: "High Stress – Significant pressure. Prioritize rest and self-care.",
        4: "Very High Stress – Overwhelming stress. Seek help and practice stress relief urgently."
    }
    return descriptions.get(level, "Unknown stress level")

if __name__ == '__main__':
    app.run(port=5001, debug=True)
