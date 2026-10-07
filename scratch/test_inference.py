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

# Load a test frame
frame = cv2.imread(r'C:\Major Project\Project-2\runs\audits\stage2_vest_target_benchmark\vis\JapanPPE.mp4_f1035.jpg')

if frame is not None:
    print("Testing processing frame...")
    for i in range(5):
        res = engine.process_frame(frame)
    print("Frame processed successfully.")
else:
    print("Could not load image, but engine initialized successfully.")
