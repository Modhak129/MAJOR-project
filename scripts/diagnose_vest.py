import cv2
import sys
import yaml
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ultralytics import YOLO

def main():
    print("Loading config...")
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    model_path = config['models']['nano']
    print(f"Loading Stage-1 model: {model_path}")
    model = YOLO(model_path)
    names = model.names
    
    PERSON_CLASS_ID = next((k for k, v in names.items() if v.lower() == "person"), 0)
    VEST_CLASS_ID = next((k for k, v in names.items() if v.lower() == "safety-vest"), 12)
    
    video_path = r"C:\Major Project\Project-2\data\raw_dataset\source_files\source_files\indianworkers.mp4"
    cap = cv2.VideoCapture(video_path)
    
    frame_idx = 0
    # Process 50 frames
    while cap.isOpened() and frame_idx < 50:
        ret, frame = cap.read()
        if not ret: break
        
        # We will sweep confidence thresholds to see when the vest drops off
        results = model.predict(frame, conf=0.05, iou=0.45, verbose=False)[0]
        
        persons = []
        vests = []
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            if cls_id == PERSON_CLASS_ID:
                persons.append(conf)
            elif cls_id == VEST_CLASS_ID:
                vests.append(conf)
                
        if len(vests) > 0:
            print(f"FRAME {frame_idx}: Highest Vest Conf = {max(vests):.3f}")
        else:
            print(f"FRAME {frame_idx}: NO VEST DETECTED EVEN AT 0.05")
        
        frame_idx += 1
        
    cap.release()

if __name__ == "__main__":
    main()
