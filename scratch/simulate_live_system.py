import yaml
import cv2
import sys
import os

# Add edge to path
sys.path.append('edge')

from inference import InferenceEngine

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize Engine
engine = InferenceEngine(config, config['models']['nano'], client_id="test_client", dashboard_url="http://localhost:5000")

# Run on a demo video
video_path = r'C:\Major Project\Project-2\data\raw_dataset\source_files\source_files\indianworkers.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open video.")
    sys.exit(1)
    
print("Starting Demo Video Validation (Smoke Test)...")
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
frames_to_process = 100
processed = 0

while processed < frames_to_process:
    ret, frame = cap.read()
    if not ret:
        break
        
    engine.process_frame(frame)
    processed += 1

print(f"Processed {processed} frames successfully.")
cap.release()
