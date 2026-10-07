import requests
import time
import uuid

dashboard_url = "http://localhost:5000"

def set_source(node_id, video_path, session_id):
    r = requests.post(f"{dashboard_url}/api/edge/source", json={
        "node_id": node_id,
        "source_type": "upload",
        "video_path": video_path,
        "session_id": session_id
    })
    return r.json()

def set_webcam(node_id):
    r = requests.post(f"{dashboard_url}/api/edge/source", json={
        "node_id": node_id,
        "source_type": "webcam"
    })
    return r.json()

def send_control(node_id, action):
    r = requests.post(f"{dashboard_url}/api/edge/demo/control", json={
        "node_id": node_id,
        "action": action
    })
    return r.json()

video_path = "data/uploads/hardhat.mp4"

print("--- MODEL SWITCH TEST ---")

# Step 1: Start on Node 1
session_1 = str(uuid.uuid4())
print("Starting on Constrained Node...")
set_source("node_constrained", video_path, session_1)
time.sleep(0.5)
send_control("node_constrained", "play")

time.sleep(5)

# Step 2: Switch to Node 2
print("Switching to Capable Node...")
set_webcam("node_constrained") # Stop old
session_2 = str(uuid.uuid4())
set_source("node_capable", video_path, session_2)
time.sleep(0.5)
send_control("node_capable", "play")

time.sleep(5)

print("Test complete.")
