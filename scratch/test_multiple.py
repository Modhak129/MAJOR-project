import yaml
import cv2
import sys
import os
import glob

# Add edge to path
sys.path.append('edge')

from inference import InferenceEngine

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize Engine
engine = InferenceEngine(config, config['models']['nano'], client_id="test_client", dashboard_url="http://localhost:5000")

images = glob.glob(r'C:\Major Project\Project-2\runs\audits\stage2_vest_target_benchmark\vis\*.jpg')

for i, img_path in enumerate(images[:50]):
    frame = cv2.imread(img_path)
    if frame is not None:
        print(f"Testing processing frame {img_path}...")
        for j in range(2):
            res = engine.process_frame(frame)
    else:
        print(f"Could not load image {img_path}.")
