import sys
import os
import cv2
import yaml
from ultralytics import YOLO

sys.path.insert(0, r"C:\Major Project\Project-2\edge")
sys.path.insert(0, r"C:\Major Project\Project-2")

def main():
    print("="*60)
    print("CONTROLLED TEST: PER-CLASS THRESHOLD VALIDATION (100 FRAMES)")
    print("="*60)

    with open(r"C:\Major Project\Project-2\config\config.yaml", "r") as f:
        config = yaml.safe_load(f)

    global_threshold = config['inference']['min_confidence']
    vest_threshold = config['inference'].get('vest_confidence', 0.35)

    model_path = r"C:\Major Project\Project-2\edge\weights\clean_sh17_yolov8n_640_best.pt"
    model = YOLO(model_path)
    
    vid_path = r"C:\Major Project\Project-2\data\raw_dataset\source_files\source_files\indianworkers.mp4"
    cap = cv2.VideoCapture(vid_path)
    
    VEST_CLASS_ID = next((k for k, v in model.names.items() if v.lower() == "safety-vest"), 12)
    
    diff_frames = 0
    
    for i in range(100):
        ret, frame = cap.read()
        if not ret: break
        
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, imgsz=320)[0]
        
        raw_confs = []
        accepted = []
        
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            if cls == VEST_CLASS_ID:
                raw_confs.append(conf)
                if conf >= vest_threshold:
                    accepted.append(conf)
                    
        # Only print frames where it made a difference (conf between 0.35 and 0.45)
        made_difference = any(vest_threshold <= c < global_threshold for c in raw_confs)
        if made_difference:
            diff_frames += 1
            print(f"Frame {i+1}")
            print(f"raw vest confidences = {[f'{c:.3f}' for c in raw_confs]}")
            print(f"global threshold = {global_threshold}")
            print(f"vest threshold = {vest_threshold}")
            print(f"accepted = {'YES' if accepted else 'NO'} ({len(accepted)} accepted)")
            print("-" * 30)

    cap.release()
    print(f"Total frames where the new threshold saved a vest detection: {diff_frames}")

if __name__ == "__main__":
    main()
