from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model
import os
import json
import datetime
import pandas as pd
from sklearn.preprocessing import minmax_scale

app = Flask(__name__)
MODEL_PATH = 'model/model.h5'
WEIGHTS_PATH = 'model/model_weights.npy'
LAST_UPDATE_FILE = 'model/last_update.txt'
CLIENT_LOG = 'model/client_logs.json'

# Load model and weights
model = load_model(MODEL_PATH)
model_weights = np.load(WEIGHTS_PATH, allow_pickle=True)
model.set_weights(model_weights)

def get_last_update():
    if os.path.exists(LAST_UPDATE_FILE):
        with open(LAST_UPDATE_FILE, 'r') as f:
            return f.read()
    return "Never"

def log_client_info(ip, device, update_time):
    log_entry = {
        "ip": ip,
        "device": device,
        "time": update_time
    }

    logs = []
    if os.path.exists(CLIENT_LOG):
        with open(CLIENT_LOG, 'r') as f:
            logs = json.load(f)

    logs.insert(0, log_entry)  # Newest first
    logs = logs[:10]  # Keep only last 10

    with open(CLIENT_LOG, 'w') as f:
        json.dump(logs, f)

@app.route('/')
def index():
    weights = model.get_weights()
    summary = [{
        "Layer": f"W{i}",
        "Shape": w.shape,
        "Min": float(np.min(w)),
        "Max": float(np.max(w)),
        "Mean": float(np.mean(w))
    } for i, w in enumerate(weights)]

    client_logs = []
    if os.path.exists(CLIENT_LOG):
        with open(CLIENT_LOG, 'r') as f:
            client_logs = json.load(f)

    centralized_acc = "Unknown"
    if os.path.exists('model/centralized_performance.txt'):
        with open('model/centralized_performance.txt', 'r') as f:
            centralized_acc = f.read().strip()

    local_accuracies = []
    accuracies = []
    if os.path.exists('model/local_accuracies.txt'):
        with open('model/local_accuracies.txt', 'r') as f:
            lines = f.readlines()
            local_accuracies = lines
            for line in lines:
                if "Local Accuracy:" in line:
                    try:
                        acc = float(line.strip().split("Local Accuracy:")[1])
                        accuracies.append(acc)
                    except:
                        pass

    if accuracies:
        global_test_acc = np.mean(accuracies)
    else:
        global_test_acc = None

    return render_template('index.html', summary=summary, last_update=get_last_update(),
                           client_logs=client_logs, centralized_acc=centralized_acc,
                           global_test_acc=global_test_acc, local_accuracies=local_accuracies)

@app.route('/get_weights', methods=['GET'])
def get_weights():
    return jsonify({f"w{i}": w.tolist() for i, w in enumerate(model.get_weights())})

@app.route('/send_update', methods=['POST'])
def receive_update():
    client_data = request.json
    client_weights = [np.array(w) for w in client_data['weights']]
    local_accuracy = client_data.get('local_accuracy', None)

    new_weights = [(w1 + w2) / 2 for w1, w2 in zip(model.get_weights(), client_weights)]
    model.set_weights(new_weights)
    model.save(MODEL_PATH)
    np.save(WEIGHTS_PATH, np.array(new_weights, dtype=object), allow_pickle=True)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LAST_UPDATE_FILE, 'w') as f:
        f.write(now)

    client_ip = request.remote_addr
    client_device = request.user_agent.string
    log_client_info(client_ip, client_device, now)

    if local_accuracy is not None:
        with open('model/local_accuracies.txt', 'a') as f:
            f.write(f"{now} - {client_ip} - Local Accuracy: {local_accuracy:.4f}\n")

    # Save dashboard data to CSV for Power BI
    global_test_acc = None
    accuracies = []
    if os.path.exists('model/local_accuracies.txt'):
        with open('model/local_accuracies.txt', 'r') as f:
            for line in f.readlines():
                if "Local Accuracy:" in line:
                    try:
                        acc = float(line.strip().split("Local Accuracy:")[1])
                        accuracies.append(acc)
                    except:
                        pass
    if accuracies:
        global_test_acc = np.mean(accuracies)

    dashboard_data = {
        "timestamp": [now],
        "client_ip": [client_ip],
        "device": [client_device],
        "local_accuracy": [local_accuracy],
        "global_accuracy": [global_test_acc if global_test_acc else 0]
    }
    df = pd.DataFrame(dashboard_data)

    csv_path = 'model/dashboard_data.csv'
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)

    return render_template("update.html", time=now)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
