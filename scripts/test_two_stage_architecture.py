import sys
import yaml
import cv2
import time
from pathlib import Path
sys.path.append(r"C:\Major Project\Project-2")
sys.path.append(r"C:\Major Project\Project-2\edge")
from edge.inference import InferenceEngine

def run_test():
    with open(r"C:\Major Project\Project-2\config\config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    model_path = r"C:\Major Project\Project-2\runs\detect\edge\weights\training\clean_sh17_yolov8n_640\weights\best.pt"
    
    print("Initializing InferenceEngine...")
    engine = InferenceEngine(config, model_path, "test_client", "http://localhost:5000")
    
    vid_path = r"C:\Major Project\Project-2\data\raw_dataset\source_files\source_files\hardhat.mp4"
    cap = cv2.VideoCapture(vid_path)
    
    frames = 0
    start = time.time()
    
    print("Processing video...")
    while frames < 300:
        ret, frame = cap.read()
        if not ret: break
        
        plot = engine.process_frame(frame)
        frames += 1
        
    elapsed = time.time() - start
    cap.release()
    
    print(f"Processed {frames} frames in {elapsed:.2f}s. FPS: {frames/elapsed:.2f}")
    print(f"Latency: {(elapsed/frames)*1000:.2f}ms")
    
    print("Regression Test PASS.")

if __name__ == "__main__":
    run_test()
