# app.py (加入 RMSE 計算並傳送)

import os
import time
import json
import joblib
import threading
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from sklearn.model_selection import train_test_split
import pandas as pd
import random

# 匯入我們的模組
from data_utils import load_and_reshape_data, create_time_series_features, normalize_data
from model import create_bp_model

app = Flask(__name__)
app.config['SECRET_KEY'] = 'a_very_secret_key!'
socketio = SocketIO(app, cors_allowed_origins="*")

training_thread = None
simulation_thread = None
is_training = threading.Event()
is_simulating = threading.Event()
soc_estimator = None

SIMULATION_STATE = {
    'speed_multiplier': 1.0,
    'v_noise': 0.0,
    'c_noise': 0.0,
    't_noise': 0.0,
    'update_interval': 0.3,
    'batch_size': 4,
}


def recalculate_simulation_timing():
    """根據速度倍率重新計算更新頻率與批次大小。"""
    multiplier = max(0.1, float(SIMULATION_STATE.get('speed_multiplier', 1.0)))
    SIMULATION_STATE['update_interval'] = max(0.05, 0.35 / multiplier)
    SIMULATION_STATE['batch_size'] = max(1, int(round(4 * multiplier)))


recalculate_simulation_timing()

class SOCEstimator:
    def __init__(self, weights_path, config_path, scaler_path):
        with open(config_path, 'r') as f:
            layer_config = json.load(f)
        self.model = create_bp_model(layer_config)
        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()
        self.scaler = joblib.load(scaler_path)
        self.window_size = layer_config[0] // 3

    def predict(self, vct_history: np.ndarray) -> float:
        if vct_history.shape[0] < self.window_size:
            return 0.0
        with torch.no_grad():
            features_flat = vct_history[-self.window_size:].flatten().reshape(1, -1)
            features_scaled = self.scaler.transform(features_flat)
            features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
            prediction_scaled = self.model(features_tensor)
            prediction = prediction_scaled.item() * 100
            return prediction

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    global soc_estimator
    print('Client connected')
    emit('initial_state', {'model_loaded': soc_estimator is not None})

@socketio.on('start_training')
def handle_start_training(params):
    global training_thread
    if is_training.is_set(): return
    is_training.set()
    training_thread = threading.Thread(target=run_training_task, args=(params,))
    training_thread.daemon = True
    training_thread.start()

@socketio.on('start_simulation')
def handle_start_simulation():
    global simulation_thread
    if not soc_estimator:
        emit('status', {'message': 'Model not available. Please train first.', 'type': 'error'})
        emit('simulation_error', {'message': '尚未載入模型，請先完成訓練。'})
        return
    if is_simulating.is_set(): return
    is_simulating.set()
    simulation_thread = threading.Thread(target=run_simulation_task)
    simulation_thread.daemon = True
    simulation_thread.start()

@socketio.on('stop_simulation')
def handle_stop_simulation():
    is_simulating.clear()

@socketio.on('update_simulation_params')
def handle_update_params(params):
    SIMULATION_STATE['speed_multiplier'] = max(0.1, float(params.get('speed', 1.0)))
    recalculate_simulation_timing()
    SIMULATION_STATE['v_noise'] = float(params.get('v_noise', 0.0))
    SIMULATION_STATE['c_noise'] = float(params.get('c_noise', 0.0))
    SIMULATION_STATE['t_noise'] = float(params.get('t_noise', 0.0))

def run_training_task(params):
    global soc_estimator
    try:
        socketio.emit('status', {'message': 'Parsing parameters...', 'type': 'info'})
        cfg = {
            'WINDOW_SIZE': int(params.get('windowSize', 5)),
            'HIDDEN_LAYERS': [int(x.strip()) for x in params.get('hiddenLayers', '64, 32').split(',')],
            'LEARNING_RATE': float(params.get('learningRate', 0.001)),
            'BATCH_SIZE': 64,
            'EPOCHS': int(params.get('epochs', 1000)),
        }
        
        socketio.emit('status', {'message': '[1/4] Loading data...', 'type': 'info'})
        train_vct_full, train_soc_full = load_and_reshape_data('data/CASE_4_Train_VCT_transposed.csv', 'data/CASE_4_Train_SOC_transposed.csv')
        
        socketio.emit('status', {'message': '[2/4] Creating features...', 'type': 'info'})
        X_full, y_full = create_time_series_features(train_vct_full, train_soc_full, cfg['WINDOW_SIZE'])
        X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
        
        socketio.emit('status', {'message': '[3/4] Normalizing data...', 'type': 'info'})
        X_train_s, X_val_s, y_train_s, y_val_s, x_scaler = normalize_data(X_train, X_val, y_train, y_val)
        
        socketio.emit('status', {'message': '[4/4] Preparing PyTorch tensors...', 'type': 'info'})
        train_loader = DataLoader(TensorDataset(torch.tensor(X_train_s, dtype=torch.float32), torch.tensor(y_train_s, dtype=torch.float32)), batch_size=cfg['BATCH_SIZE'], shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.tensor(X_val_s, dtype=torch.float32), torch.tensor(y_val_s, dtype=torch.float32)), batch_size=cfg['BATCH_SIZE'])

        input_size = X_train.shape[1]
        layer_config = [input_size] + cfg['HIDDEN_LAYERS'] + [1]
        model = create_bp_model(layer_config)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=cfg['LEARNING_RATE'])

        socketio.emit('status', {'message': 'Starting training loop...', 'type': 'training'})
        for epoch in range(cfg['EPOCHS']):
            if not is_training.is_set(): break
            
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            model.eval()
            val_losses, all_preds, all_true = [], [], []
            with torch.no_grad():
                for batch_X_val, batch_y_val in val_loader:
                    val_outputs = model(batch_X_val)
                    val_losses.append(criterion(val_outputs, batch_y_val).item())
                    all_preds.append(val_outputs.numpy())
                    all_true.append(batch_y_val.numpy())
            
            epoch_val_loss = np.mean(val_losses)

            # --- MODIFIED: 計算並加入 RMSE ---
            preds_rescaled = np.concatenate(all_preds) * 100
            true_rescaled = np.concatenate(all_true) * 100
            val_rmse = np.sqrt(np.mean((preds_rescaled - true_rescaled)**2))
            
            socketio.emit('training_update', {
                'epoch': epoch + 1, 'total_epochs': cfg['EPOCHS'],
                'train_loss': float(loss.item()), 
                'val_loss': float(epoch_val_loss),
                'val_rmse': float(val_rmse) # <--- NEW: 加入 RMSE 數據
            })
            socketio.sleep(0.01)

        os.makedirs('saved_model', exist_ok=True)
        weights_path, scaler_path, config_path = 'saved_model/soc_model_weights.pth', 'saved_model/x_scaler.pkl', 'saved_model/model_config.json'
        torch.save(model.state_dict(), weights_path)
        joblib.dump(x_scaler, scaler_path)
        with open(config_path, 'w') as f: json.dump(layer_config, f)
        
        soc_estimator = SOCEstimator(weights_path, config_path, scaler_path)
        socketio.emit('status', {'message': 'Training completed successfully.', 'type': 'success'})
        socketio.emit('training_finished', {'model_loaded': True})

    except Exception as e:
        socketio.emit('status', {'message': f'Error: {str(e)}', 'type': 'error'})
    finally:
        is_training.clear()

def run_simulation_task():
    global soc_estimator
    if not soc_estimator:
        socketio.emit('simulation_error', {'message': '尚未載入模型，請先完成訓練。'})
        return

    socketio.emit('status', {'message': 'Simulation running...', 'type': 'simulating'})

    try:
        vct_sim_data, soc_sim_data = load_and_reshape_data(
            'data/CASE_4_Test_VCT_transposed.csv', 'data/CASE_4_Test_SOC_transposed.csv'
        )
        soc_series = soc_sim_data.flatten()
        total_points = len(vct_sim_data)

        socketio.emit('simulation_started', {
            'total_points': int(total_points),
            'window_size': int(soc_estimator.window_size),
            'speed_multiplier': SIMULATION_STATE['speed_multiplier'],
            'update_interval': SIMULATION_STATE['update_interval'],
            'batch_size': SIMULATION_STATE['batch_size'],
        })

        history = []
        index = 0
        update_batch = []
        last_emit_time = time.time()

        while is_simulating.is_set():
            row_idx = index % total_points
            original_row = vct_sim_data[row_idx]
            noisy_row = [
                original_row[0] + SIMULATION_STATE['v_noise'] * random.uniform(-0.5, 0.5),
                original_row[1] + SIMULATION_STATE['c_noise'] * random.uniform(-0.5, 0.5),
                original_row[2] + SIMULATION_STATE['t_noise'] * random.uniform(-0.5, 0.5),
            ]

            history.append(noisy_row)
            if len(history) > soc_estimator.window_size:
                history.pop(0)

            predicted_soc = soc_estimator.predict(np.array(history))
            actual_soc = float(soc_series[row_idx])
            error = predicted_soc - actual_soc

            update_batch.append({
                'index': index,
                'v': noisy_row[0],
                'c': noisy_row[1],
                't': noisy_row[2],
                'soc': predicted_soc,
                'actual_soc': actual_soc,
                'error': error,
            })

            now = time.time()
            emit_due_to_size = len(update_batch) >= SIMULATION_STATE['batch_size']
            emit_due_to_time = now - last_emit_time >= SIMULATION_STATE['update_interval']

            if emit_due_to_size or emit_due_to_time:
                socketio.emit('simulation_update', {
                    'points': update_batch,
                    'server_ts': now,
                    'latency_hint_ms': int((now - last_emit_time) * 1000),
                    'speed_multiplier': SIMULATION_STATE['speed_multiplier'],
                })
                update_batch = []
                last_emit_time = now

            index += 1
            sleep_duration = SIMULATION_STATE['update_interval'] / max(SIMULATION_STATE['batch_size'], 1)
            socketio.sleep(max(0.01, sleep_duration))

        if update_batch:
            now = time.time()
            socketio.emit('simulation_update', {
                'points': update_batch,
                'server_ts': now,
                'latency_hint_ms': int((now - last_emit_time) * 1000),
                'speed_multiplier': SIMULATION_STATE['speed_multiplier'],
            })

    except Exception as exc:
        socketio.emit('simulation_error', {'message': f'模擬程序發生錯誤：{exc}'})
    finally:
        is_simulating.clear()
        socketio.emit('status', {'message': 'Simulation stopped.', 'type': 'info'})
        socketio.emit('simulation_stopped', {'timestamp': time.time()})

if __name__ == '__main__':
    weights_p, config_p, scaler_p = 'saved_model/soc_model_weights.pth', 'saved_model/model_config.json', 'saved_model/x_scaler.pkl'
    if all(os.path.exists(p) for p in [weights_p, config_p, scaler_p]):
        print("Found existing trained model. Loading...")
        try:
            soc_estimator = SOCEstimator(weights_p, config_p, scaler_p)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading pre-trained model: {e}")
    else:
        print("No pre-trained model found. Please train a model via the web UI.")

    print("Starting server... Please open http://127.0.0.1:5000 in your browser.")
    socketio.run(app, debug=False, allow_unsafe_werkzeug=True)