import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import flwr as fl
import os
import yaml
import torch

# Prevent PyTorch from spawning massive threads leading to CPU thrashing
torch.set_num_threads(2)
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np
from ultralytics import YOLO
from collections import OrderedDict
import requests
import threading
import time
import sys
import base64
import cv2
import socketio
from datetime import datetime
sys.path.append('/app/edge')
from inference import InferenceEngine

os.environ["YOLO_VERBOSE"] = "False"

sio = socketio.Client()
inference_engine = None
inference_running = True
training_running = True
manual_training_trigger = False
demo_training_trigger = False
edge_mode = "MONITOR"

class YOLOClient(fl.client.NumPyClient):
    def __init__(self, config, client_id, model_tier, data_yaml):
        self.config = config
        self.client_id = client_id
        self.model_tier = model_tier
        self.model_path = config['models'][model_tier]
        
        model_name = "yolov8n.pt" if model_tier == "nano" else "yolov8s.pt"
        self.model = YOLO(self.model_path)
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save({'model': self.model.model.half()}, self.model_path)
        self.data_yaml = data_yaml
        
        # EIM-07: Post-Training Quantization (PTQ INT8)
        if model_tier == "nano":
            try:
                self.model.export(format="onnx", int8=True)
                print(f"[{self.client_id}] Successfully exported INT8 ONNX model.")
            except Exception as e:
                print(f"[{self.client_id}] INT8 export failed/skipped: {e}")
        
    def get_properties(self, config):
        import multiprocessing
        return {
            "vram_mb": int(os.environ.get("SIMULATED_VRAM_MB", 2048)),
            "cpu_cores": multiprocessing.cpu_count(),
            "client_id": self.client_id
        }

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.model.state_dict().items()]

    def set_parameters(self, parameters):
        if not parameters:
            return
        params_dict = zip(self.model.model.state_dict().keys(), parameters)
        
        # Safely filter out size mismatches for Adaptive Tier FL
        current_state = self.model.model.state_dict()
        valid_state_dict = OrderedDict()
        for k, v in params_dict:
            t = torch.tensor(v)
            if k in current_state and current_state[k].shape == t.shape:
                valid_state_dict[k] = t
            else:
                # Shape mismatch due to Nano vs Small tier, keep original
                valid_state_dict[k] = current_state[k]
                
        self.model.model.load_state_dict(valid_state_dict, strict=False)
        torch.save({'model': self.model.model.half()}, self.model_path)
        
        global inference_engine
        if inference_engine:
            inference_engine.reload_model(self.model_path)
            
        dashboard_url = os.environ.get("DASHBOARD_URL", "http://dashboard:5000")
        try:
            requests.post(f"{dashboard_url}/api/telemetry", json={
                "node_id": self.client_id,
                "latest_model_received_time": datetime.utcnow().isoformat() + "Z"
            })
        except:
            pass

    def fit(self, parameters, config):
        global manual_training_trigger, edge_mode, demo_training_trigger, inference_running
        
        while edge_mode == "MONITOR" and not manual_training_trigger and not demo_training_trigger:
            time.sleep(2)
        
        self.set_parameters(parameters)
        
        if edge_mode == "DEMO_TRAIN" or demo_training_trigger:
            demo_training_trigger = False
            print(f"[{self.client_id}] Running SUPER FAST DEMO_TRAIN (Simulated)...")
            # Skip YOLO engine initialization for demo purposes to instantly trigger animation
            time.sleep(4)
            edge_mode = "MONITOR"
            inference_running = True
            requests.post(f"{dashboard_url}/api/telemetry", json={"node_id": self.client_id, "status": "MONITOR"})
            
        elif edge_mode == "TRAIN" or manual_training_trigger:
            manual_training_trigger = False
            epochs = self.config['fl']['local_epochs']
            batch_size = self.config['fl']['batch_size']
            
            target_data = self.data_yaml
            import sys
            import yaml
            if sys.platform == 'win32':
                with open(self.data_yaml, 'r') as f:
                    yd = yaml.safe_load(f)
                yd['path'] = os.path.abspath('data')
                target_data = self.data_yaml.replace('.yaml', '_win.yaml')
                with open(target_data, 'w') as f:
                    yaml.dump(yd, f)
            if os.path.exists("data/local_training/images") and len(os.listdir("data/local_training/images")) > 0:
                print(f"[{self.client_id}] Using cached violation frames for training!")
                local_yaml = "data/local_training/dataset.yaml"
                with open(local_yaml, "w") as f:
                    f.write(f"path: {os.path.abspath('data/local_training')}\ntrain: images\nval: images\nnames: {{0: 'ppe', 1: 'fall', 2: 'zone'}}")
                target_data = local_yaml
                
            print(f"[{self.client_id}] Starting local training for {epochs} epochs...")
            import sys
            device = '0' if sys.platform == 'win32' else 'cpu'
            self.model.train(
                data=target_data,
                epochs=epochs,
                batch=batch_size,
                imgsz=self.config['inference']['input_resolution'],
                device=device,
                workers=0
            )
            
        edge_mode = "MONITOR"
        inference_running = True
        dashboard_url = os.environ.get("DASHBOARD_URL", "http://dashboard:5000")
        try:
            requests.post(f"{dashboard_url}/api/telemetry", json={"node_id": self.client_id, "status": "MONITOR"})
        except: pass
        
        updated_weights = self.get_parameters(config)
        k = self.config['adaptive']['sparsification_k']
        sparse_weights = []
        for w in updated_weights:
            shape = w.shape
            w_flat = w.flatten()
            num_top_k = max(1, int(len(w_flat) * k))
            indices = np.argpartition(np.abs(w_flat), -num_top_k)[-num_top_k:]
            mask = np.zeros_like(w_flat, dtype=bool)
            mask[indices] = True
            w_sparse = np.where(mask, w_flat, 0)
            sparse_weights.append(w_sparse.reshape(shape))
            
        print(f"[{self.client_id}] Finished training. Sending sparsified weights.")
        return sparse_weights, len(self.model.model.state_dict()), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        import sys
        device = '0' if sys.platform == 'win32' else 'cpu'
        
        target_data = self.data_yaml
        if sys.platform == 'win32':
            import yaml
            with open(self.data_yaml, 'r') as f:
                yd = yaml.safe_load(f)
            yd['path'] = os.path.abspath('data')
            target_data = self.data_yaml.replace('.yaml', '_win.yaml')
            with open(target_data, 'w') as f:
                yaml.dump(yd, f)
                
        metrics = self.model.val(data=target_data, device=device)
        map_val = float(metrics.box.map) if metrics.box.map is not None else 0.0
        return map_val, len(self.model.model.state_dict()), {"map": map_val}

# --- SocketIO Handlers ---
@sio.on('connect')
def on_connect():
    print('Connected to Dashboard SocketIO')

process_lock = threading.Lock()

@sio.on('process_frame')
def on_process_frame(data):
    global inference_engine, inference_running, edge_mode, video_source_type
    
    if not process_lock.acquire(blocking=False):
        # Drop frame to prevent thread explosion and OOM
        return
        
    try:
        target_node = data.get("target_node")
        client_id = os.environ.get("CLIENT_ID", "unknown")
        if target_node and target_node != client_id:
            return
            
        if not inference_running or edge_mode != "MONITOR" or video_source_type == "upload":
            return

        # Decode base64 frame
        b64_img = data.get("image", "")
        img_data = base64.b64decode(b64_img.split(',')[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if inference_engine is None:
            return
            
        # Inference
        processed = inference_engine.process_frame(frame)
        
        # Encode back to base64
        _, buffer = cv2.imencode('.jpg', processed)
        b64_str = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
        
        # Send back
        sio.emit('processed_frame_result', b64_str)
    except Exception as e:
        print(f"Frame processing error: {e}", flush=True)
    finally:
        process_lock.release()

@sio.on('update_geofence')
def on_update_geofence(data):
    global inference_engine
    if inference_engine:
        inference_engine.update_geofence(data.get("polygon", []))

# Video Demo State
video_source_type = "webcam"
demo_playback_state = "stopped"
demo_video_path = None
demo_cap = None
demo_frame_delay = 0.033

demo_session_id = None

@sio.on('edge_control')
def on_edge_control(data):
    global inference_running, manual_training_trigger, edge_mode, demo_training_trigger
    global video_source_type, demo_playback_state, demo_video_path, demo_cap, demo_frame_delay
    global demo_session_id
    client_id = os.environ.get("CLIENT_ID", "unknown")
    if data.get("client_id") == client_id:
        cmd = data.get("command")
        mode = data.get("mode")
        
        if cmd == "mode_switch":
            edge_mode = mode
            if mode == "MONITOR":
                inference_running = True
                manual_training_trigger = False
                demo_training_trigger = False
            elif mode == "TRAIN":
                inference_running = False
                manual_training_trigger = True
            elif mode == "DEMO_TRAIN":
                inference_running = False
                demo_training_trigger = True
                
            dashboard_url = os.environ.get("DASHBOARD_URL", "http://dashboard:5000")
            try:
                requests.post(f"{dashboard_url}/api/telemetry", json={"node_id": client_id, "status": mode})
            except:
                pass
            print(f"[{client_id}] Control command applied: {mode}")

        elif cmd == "set_source":
            source_data = data.get("source", {})
            src_type = source_data.get("source_type")
            if src_type == "webcam":
                video_source_type = "webcam"
                demo_playback_state = "stopped"
                if demo_cap:
                    demo_cap.release()
                    demo_cap = None
                print(f"[{client_id}] Source set to: webcam")
            elif src_type == "upload":
                # We do NOT immediately open the video. We wait for 'play' or 'reset' to actually start.
                video_source_type = "upload"
                demo_video_path = source_data.get("video_path")
                demo_playback_state = "ready"
                if demo_cap:
                    demo_cap.release()
                    demo_cap = None
                demo_session_id = source_data.get("session_id", "unknown")
                print(f"[{client_id}] Source set to: upload, session: {demo_session_id}")

        elif cmd == "demo_control":
            action = data.get("action")
            if video_source_type != "upload":
                return
                
            if action == "play":
                if demo_playback_state in ["ready", "stopped", "finished"]:
                    # Start fresh
                    if demo_cap:
                        demo_cap.release()
                    demo_cap = cv2.VideoCapture(demo_video_path)
                    fps = demo_cap.get(cv2.CAP_PROP_FPS)
                    if fps <= 0: fps = 30
                    demo_frame_delay = 1.0 / fps
                    if inference_engine:
                        inference_engine.reset_tracker()
                demo_playback_state = "playing"
                
            elif action == "pause":
                demo_playback_state = "paused"
                
            elif action == "reset":
                if demo_cap:
                    demo_cap.release()
                demo_cap = cv2.VideoCapture(demo_video_path)
                if inference_engine:
                    inference_engine.reset_tracker()
                demo_playback_state = "paused"
                
            print(f"[{client_id}] Demo control: {action}, state: {demo_playback_state}")

def run_background_inference_loop():
    global inference_engine, inference_running, edge_mode
    global video_source_type, demo_playback_state, demo_cap, demo_frame_delay
    global demo_session_id
    
    first_frame_emitted = False
    
    while True:
        if not inference_running or edge_mode != "MONITOR" or video_source_type != "upload":
            time.sleep(0.5)
            first_frame_emitted = False
            continue
            
        if demo_playback_state != "playing":
            time.sleep(0.1)
            continue
            
        if demo_cap is None or not demo_cap.isOpened():
            print(f"Demo error: cap not opened")
            demo_playback_state = "error"
            sio.emit('demo_error', {"node_id": os.environ.get("CLIENT_ID"), "session_id": demo_session_id})
            continue
            
        start_time = time.time()
        ret, frame = demo_cap.read()
        
        if not ret:
            print(f"Demo finished (EOF)")
            demo_playback_state = "finished"
            client_id = os.environ.get("CLIENT_ID", "unknown")
            sio.emit('demo_finished', {"node_id": client_id, "session_id": demo_session_id})
            if demo_cap:
                demo_cap.release()
                demo_cap = None
            first_frame_emitted = False
            continue
            
        try:
            processed = inference_engine.process_frame(frame)
            _, buffer = cv2.imencode('.jpg', processed)
            b64_str = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
            sio.emit('processed_frame_result', {"image": b64_str, "session_id": demo_session_id})
            
            if not first_frame_emitted:
                sio.emit('demo_started', {"node_id": os.environ.get("CLIENT_ID"), "session_id": demo_session_id})
                first_frame_emitted = True
                
        except Exception as e:
            print(f"Demo processing error: {e}", flush=True)
            
        elapsed = time.time() - start_time
        sleep_time = max(0, demo_frame_delay - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

def main():
    global inference_engine
    
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    client_id = os.environ.get("CLIENT_ID", "unknown")
    vram_mb = int(os.environ.get("SIMULATED_VRAM_MB", 2048))
    server_address = os.environ.get("SERVER_ADDRESS", "fl_server:8080")
    dashboard_url = os.environ.get("DASHBOARD_URL", "http://dashboard:5000")
    
    tier = "nano" if vram_mb < config['adaptive']['vram_threshold_mb'] else "small"
    print(f"[{client_id}] Assigned tier: {tier}")
    
    # Initialize Engine
    model_path = config['models'][tier]
    
    # Connect to SocketIO
    connected = False
    while not connected:
        try:
            sio.connect(dashboard_url)
            connected = True
        except Exception as e:
            print(f"SocketIO connect failed: {e}. Retrying in 5 seconds...", flush=True)
            time.sleep(5)

    client_num = 3 if "native_gpu" in client_id else (1 if "constrained" in client_id else 2)
    data_yaml = f"data/partitioned/client_{client_num}/dataset.yaml"
    client = YOLOClient(config, client_id, tier, data_yaml)
    
    while not os.path.exists(model_path):
        time.sleep(1)
        
    inference_engine = InferenceEngine(config, model_path, client_id, dashboard_url)
    
    # Fetch initial geofence via REST
    try:
        res = requests.get(f"{dashboard_url}/api/geofence")
        inference_engine.update_geofence(res.json().get("polygon", []))
    except:
        pass

    threading.Thread(target=run_background_inference_loop, daemon=True).start()
    
    time.sleep(15) # Wait for server
    
    try:
        requests.post(f"{dashboard_url}/api/clients", json={"client_id": client_id, "tier": tier, "status": "Training"})
        fl.client.start_client(server_address=server_address, client=client.to_client())
        requests.post(f"{dashboard_url}/api/clients", json={"client_id": client_id, "tier": tier, "status": "Completed"})
    except Exception as e:
        print(f"[{client_id}] FL client failed: {e}")
        
    while True:
        time.sleep(60)

if __name__ == "__main__":
    main()
