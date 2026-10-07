import sys
import os
import cv2
from ultralytics import YOLO

sys.path.insert(0, r"C:\Major Project\Project-2\edge")
sys.path.insert(0, r"C:\Major Project\Project-2")

def main():
    model_path = r"C:\Major Project\Project-2\edge\weights\clean_sh17_yolov8n_640_best.pt"
    model = YOLO(model_path)
    
    videos = [
        r"C:\Major Project\Project-2\data\raw_dataset\source_files\source_files\hardhat.mp4",
        r"C:\Major Project\Project-2\data\raw_dataset\source_files\source_files\indianworkers.mp4",
        r"C:\Major Project\Project-2\data\raw_dataset\source_files\source_files\JapanPPE.mp4"
    ]
    
    for vid_path in videos:
        print(f"\nAnalyzing: {os.path.basename(vid_path)}")
        cap = cv2.VideoCapture(vid_path)
        
        vest_confs = []
        for i in range(20):
            ret, frame = cap.read()
            if not ret: break
            results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, imgsz=320)[0]
            for box in results.boxes:
                if int(box.cls[0]) == 12: # Safety-vest
                    vest_confs.append(float(box.conf[0]))
                    
        cap.release()
        
        if not vest_confs:
            print("  No vests detected.")
        else:
            print(f"  Vests detected: {len(vest_confs)}")
            print(f"  Min: {min(vest_confs):.3f}, Max: {max(vest_confs):.3f}, Mean: {sum(vest_confs)/len(vest_confs):.3f}")
            print(f"  >= 0.45: {sum(1 for c in vest_confs if c >= 0.45)}")
            print(f"  >= 0.40: {sum(1 for c in vest_confs if c >= 0.40)}")
            print(f"  >= 0.35: {sum(1 for c in vest_confs if c >= 0.35)}")
            print(f"  >= 0.30: {sum(1 for c in vest_confs if c >= 0.30)}")
            print(f"  >= 0.25: {sum(1 for c in vest_confs if c >= 0.25)}")
            print(f"  >= 0.20: {sum(1 for c in vest_confs if c >= 0.20)}")

if __name__ == "__main__":
    main()
