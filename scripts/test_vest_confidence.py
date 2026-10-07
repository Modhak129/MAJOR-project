import sys
import os
import cv2
import yaml
import numpy as np

sys.path.insert(0, r"C:\Major Project\Project-2\edge")
sys.path.insert(0, r"C:\Major Project\Project-2")
os.chdir(r"C:\Major Project\Project-2")

from inference import InferenceEngine
from ultralytics import YOLO

def main():
    print("="*60)
    print("STEP 4 & 5: INVESTIGATE SAFETY VEST CONFIDENCE")
    print("="*60)
    
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # We will test safety-vest across different thresholds without modifying config.yaml yet.
    # First, let's load the exact model to inspect raw confidences for safety-vest
    model_path = r"C:\Major Project\Project-2\edge\weights\clean_sh17_yolov8n_640_best.pt"
    model = YOLO(model_path)
    
    vid_path = r"C:\Major Project\Project-2\data\raw_dataset\source_files\source_files\indianworkers.mp4"
    cap = cv2.VideoCapture(vid_path)
    
    vest_confs = []
    
    for i in range(20):
        ret, frame = cap.read()
        if not ret: break
        
        # Run YOLO with very low confidence to capture all vest detections
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, imgsz=320)[0]
        
        frame_vest_confs = []
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            # Safety-vest is class 12
            if cls == 12:
                frame_vest_confs.append(conf)
                
        vest_confs.extend(frame_vest_confs)
        print(f"Frame {i+1} vest confidences: {[f'{c:.3f}' for c in frame_vest_confs]}")
        
    cap.release()
    
    if not vest_confs:
        print("No vests detected at all.")
        return
        
    print(f"\nVest Confidence Stats over 20 frames:")
    print(f"Mean: {np.mean(vest_confs):.3f}")
    print(f"Max: {np.max(vest_confs):.3f}")
    print(f"Min: {np.min(vest_confs):.3f}")
    
    print(f"\nDetections >= 0.45: {sum(1 for c in vest_confs if c >= 0.45)}")
    print(f"Detections >= 0.40: {sum(1 for c in vest_confs if c >= 0.40)}")
    print(f"Detections >= 0.35: {sum(1 for c in vest_confs if c >= 0.35)}")
    print(f"Detections >= 0.30: {sum(1 for c in vest_confs if c >= 0.30)}")
    print(f"Detections >= 0.25: {sum(1 for c in vest_confs if c >= 0.25)}")

if __name__ == "__main__":
    main()
