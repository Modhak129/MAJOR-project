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
    HELMET_CLASS_ID = next((k for k, v in names.items() if v.lower() == "helmet"), 14)
    VEST_CLASS_ID = next((k for k, v in names.items() if v.lower() == "safety-vest"), 12)
    GLOVES_CLASS_ID = next((k for k, v in names.items() if v.lower() == "gloves"), 9)
    
    print(f"Class IDs: Person={PERSON_CLASS_ID}, Helmet={HELMET_CLASS_ID}, Vest={VEST_CLASS_ID}, Gloves={GLOVES_CLASS_ID}")
    
    video_path = r"C:\Major Project\Project-2\data\raw_dataset\source_files\source_files\hardhat.mp4"
    cap = cv2.VideoCapture(video_path)
    
    # We will look at frame 50 as a sample, or we can just scan until we see two people.
    # To be thorough, let's process the first 60 frames and log the RAW detections.
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Raw inference with very low confidence
        results = model.predict(frame, conf=0.05, iou=0.45, verbose=False)[0]
        
        persons = []
        vests = []
        helmets = []
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            
            if cls_id == PERSON_CLASS_ID:
                persons.append({"conf": conf, "box": xyxy})
            elif cls_id == VEST_CLASS_ID:
                vests.append({"conf": conf, "box": xyxy})
            elif cls_id == HELMET_CLASS_ID:
                helmets.append({"conf": conf, "box": xyxy})
                
        if len(persons) > 1 or len(vests) > 0:
            print(f"\n--- FRAME {frame_idx} ---")
            print("PERSONS:")
            persons = sorted(persons, key=lambda p: p["box"][0])
            for i, p in enumerate(persons):
                x1, y1, x2, y2 = p["box"]
                print(f"  Worker {i} (x={x1:.1f}): conf={p['conf']:.3f}, w={x2-x1:.1f}, h={y2-y1:.1f}")
                
            print("VESTS:")
            for i, v in enumerate(vests):
                print(f"  Vest {i}: conf={v['conf']:.3f}")
                
            print("HELMETS:")
            for i, h in enumerate(helmets):
                print(f"  Helmet {i}: conf={h['conf']:.3f}")
        
        frame_idx += 1
        
    cap.release()

if __name__ == "__main__":
    main()
