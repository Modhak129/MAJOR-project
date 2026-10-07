import yaml
import cv2
import time
import sys
import os
sys.path.append(os.path.abspath('edge'))
from edge.inference import InferenceEngine

def test_mode(mode_name):
    print(f"\n======================================")
    print(f"TESTING {mode_name.upper()} MODE")
    print(f"======================================")
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    config['inference']['ppe_detector_mode'] = mode_name
    
    engine = InferenceEngine(config, "edge/weights/sh17_yolov8n_best.pt", "test_client", "http://localhost:5000")
    
    cap = cv2.VideoCapture("data/raw_dataset/source_files/source_files/JapanPPE.mp4")
    
    print("Processing 5 frames...")
    # Fast forward to frame 287 where person is visible
    cap.set(cv2.CAP_PROP_POS_FRAMES, 287)
    
    for i in range(5):
        ret, frame = cap.read()
        if not ret: break
        
        t0 = time.time()
        out = engine.process_frame(frame)
        t1 = time.time()
        
        print(f"Frame {i+1} processed in {(t1-t0)*1000:.1f}ms")
    
    cap.release()
    print("Test completed.")

if __name__ == "__main__":
    test_mode("sh17")
