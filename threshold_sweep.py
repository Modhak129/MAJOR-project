import cv2
import sys
import os
import time
sys.path.append(os.path.abspath('edge'))
from ultralytics import YOLO

def sweep(model_path, model_name, class_map, test_frames):
    print(f"\n--- Sweeping {model_name} ---")
    model = YOLO(model_path)
    
    thresholds = [0.05, 0.10, 0.20, 0.30, 0.40, 0.45, 0.50, 0.60, 0.70]
    
    table = "| Threshold | Persons | Helmets (Raw) | Vests (Raw) | Gloves (Raw) |\n"
    table += "|---|---|---|---|---|\n"
    
    for conf in thresholds:
        persons = 0
        helmets = 0
        vests = 0
        gloves = 0
        
        for frame in test_frames:
            res = model(frame, imgsz=640, device=0, conf=conf, verbose=False)[0]
            for box in res.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in class_map: continue
                semantic_cls = class_map[cls_id]
                
                if semantic_cls == "person": persons += 1
                elif semantic_cls == "helmet": helmets += 1
                elif semantic_cls == "safety_vest": vests += 1
                elif semantic_cls == "gloves": gloves += 1
                
        table += f"| {conf:.2f} | {persons} | {helmets} | {vests} | {gloves} |\n"
        
    return table

def run_sweep():
    video_path = os.path.abspath(r'data\raw_dataset\source_files\source_files\JapanPPE.mp4')
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 287)
    
    test_frames = []
    while len(test_frames) < 50:
        ret, frame = cap.read()
        if not ret: break
        test_frames.append(frame)
    cap.release()
    
    sh17_map = {0: "person", 14: "helmet", 12: "safety_vest", 9: "gloves"}
    rf_map = {11: "person", 3: "helmet", 13: "safety_vest", 1: "gloves"}
    
    sh17_table = sweep(r'edge\weights\sh17_yolov8n_best.pt', 'SH17', sh17_map, test_frames)
    rf_table = sweep(r'runs\detect\edge\weights\training\roboflow\roboflow_yolov8n_640\weights\best.pt', 'Roboflow', rf_map, test_frames)
    
    out = "# Threshold Sweep Results\n\n## SH17 YOLOv8n\n" + sh17_table + "\n## Roboflow YOLOv8n\n" + rf_table
    with open("results/roboflow_vs_sh17/threshold_sweep.md", "w") as f:
        f.write(out)
        
    print("\n" + out)

if __name__ == "__main__":
    run_sweep()
