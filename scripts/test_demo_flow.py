import requests
import time
import uuid

dashboard_url = "http://localhost:5000"

def upload_video():
    # Use hardhat.mp4 for test
    with open(r"C:\Major Project\Project-2\data\raw_dataset\source_files\source_files\hardhat.mp4", 'rb') as f:
        r = requests.post(f"{dashboard_url}/api/video/upload", files={"video": f})
        return r.json()

def set_source(node_id, video_path, session_id):
    r = requests.post(f"{dashboard_url}/api/edge/source", json={
        "node_id": node_id,
        "source_type": "upload",
        "video_path": video_path,
        "session_id": session_id
    })
    return r.json()

def send_control(node_id, action):
    r = requests.post(f"{dashboard_url}/api/edge/demo/control", json={
        "node_id": node_id,
        "action": action
    })
    return r.json()

print("Uploading...")
res = upload_video()
video_path = res["file_path"]

print(f"Video uploaded: {video_path}")

session_id = str(uuid.uuid4())
print(f"Session ID: {session_id}")

print("Setting source...")
set_source("node_constrained", video_path, session_id)
time.sleep(1)

print("Starting playback...")
send_control("node_constrained", "play")

print("Waiting 10 seconds for frames...")
time.sleep(10)

print("Pausing...")
send_control("node_constrained", "pause")
time.sleep(2)

print("Resuming...")
send_control("node_constrained", "play")
time.sleep(5)

print("Done scripting.")
