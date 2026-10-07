import sys
import yaml
import cv2
import time
from pathlib import Path
sys.path.append(r"C:\Major Project\Project-2")
sys.path.append(r"C:\Major Project\Project-2\edge")
from edge.inference import InferenceEngine

with open(r"C:\Major Project\Project-2\config\config.yaml", 'r') as f:
    config = yaml.safe_load(f)

model_path = r"C:\Major Project\Project-2\runs\detect\edge\weights\training\clean_sh17_yolov8n_640\weights\best.pt"
engine = InferenceEngine(config, model_path, "test_client", "http://localhost:5000")

img_path = r"C:\Users\Vishnu\.gemini\antigravity\brain\3a395107-a56b-4df5-a7db-327a4da905c5\.user_uploaded\media_1791024184799.png"
full_img = cv2.imread(img_path)
H, W = full_img.shape[:2]
frame = full_img[0:int(H*0.65), :]

# Process frame
engine.process_frame(frame)
print("Finished processing frame.")
