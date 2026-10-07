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
    
    total_frames = 0
    fp_vests = 0
    
    for vid_path in videos:
        cap = cv2.VideoCapture(vid_path)
        for i in range(50):  # Check more frames
            ret, frame = cap.read()
            if not ret: break
            total_frames += 1
            
            results = model.predict(frame, conf=0.35, imgsz=320, verbose=False)[0]
            vests = [box for box in results.boxes if int(box.cls[0]) == 12]
            
            # Simple heuristic: if it's hardhat.mp4 or JapanPPE (which I noticed earlier had no vests in 20 frames), 
            # let's just see if ANY vests pop up at 0.35 in 50 frames.
            if "hardhat" in vid_path or "JapanPPE" in vid_path:
                fp_vests += len(vests)
                
        cap.release()
        
    print(f"Total false-positive vests at 0.30 conf on hardhat/JapanPPE over 100 frames: {fp_vests}")

if __name__ == "__main__":
    main()
