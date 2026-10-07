import cv2
import time
import sys
import os
import json
import numpy as np
sys.path.append(os.path.abspath('edge'))
from ultralytics import YOLO

def extract_metrics(model_path, model_name, class_map, test_frames):
    print(f"\n--- Running benchmark for {model_name} ---")
    model = YOLO(model_path)
    
    metrics = {
        "model": model_name,
        "total_persons": 0,
        "total_helmets": 0,
        "total_vests": 0,
        "total_gloves": 0,
        "unique_workers": set(),
        "latency_ms": 0.0,
        "fps": 0.0
    }
    
    t0 = time.time()
    
    frame_idx = 0
    for frame in test_frames:
        frame_idx += 1
        res = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, imgsz=640, device=0, conf=0.45)[0]
        
        persons = []
        items = []
        
        # Save a few frames for visual comparison
        if frame_idx % 10 == 0 or frame_idx == 1:
            annotated = res.plot()
            os.makedirs("results/roboflow_vs_sh17/visuals", exist_ok=True)
            cv2.imwrite(f"results/roboflow_vs_sh17/visuals/frame_{(287+frame_idx-1):04d}_{model_name}.jpg", annotated)
        
        for box in res.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            wid = int(box.id[0]) if box.id is not None else -1
            if conf < 0.45: continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            if cls_id not in class_map: continue
            
            semantic_cls = class_map[cls_id]
            
            if semantic_cls == "person":
                persons.append({'box': [x1, y1, x2, y2], 'id': wid})
            else:
                items.append({'cls': semantic_cls, 'box': [x1, y1, x2, y2]})
                
        metrics["total_persons"] += len(persons)
        
        for person in persons:
            px1, py1, px2, py2 = person['box']
            metrics["unique_workers"].add(person['id'])
            
            found_items = set()
            for item in items:
                ix1, iy1, ix2, iy2 = item['box']
                inter_x1 = max(px1, ix1)
                inter_y1 = max(py1, iy1)
                inter_x2 = min(px2, ix2)
                inter_y2 = min(py2, iy2)
                
                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    item_area = (ix2 - ix1) * (iy2 - iy1)
                    if inter_area / item_area > 0.5:
                        found_items.add(item['cls'])
                        
            if 'helmet' in found_items: metrics["total_helmets"] += 1
            if 'safety_vest' in found_items: metrics["total_vests"] += 1
            if 'gloves' in found_items: metrics["total_gloves"] += 1
            
        metrics["raw_helmets"] = metrics.get("raw_helmets", 0) + sum(1 for i in items if i['cls'] == 'helmet')
        metrics["raw_vests"] = metrics.get("raw_vests", 0) + sum(1 for i in items if i['cls'] == 'safety_vest')
        metrics["raw_gloves"] = metrics.get("raw_gloves", 0) + sum(1 for i in items if i['cls'] == 'gloves')

    t1 = time.time()
    metrics["latency_ms"] = (t1 - t0) * 1000 / len(test_frames)
    metrics["fps"] = 1000 / metrics["latency_ms"]
    metrics["unique_workers"] = len(metrics["unique_workers"])
    
    return metrics

def run_benchmark():
    video_path = os.path.abspath(r'data\raw_dataset\source_files\source_files\JapanPPE.mp4')
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 287)
    
    test_frames = []
    while len(test_frames) < 50:
        ret, frame = cap.read()
        if not ret: break
        test_frames.append(frame)
    cap.release()
    print(f"Captured {len(test_frames)} frames for A/B benchmark.")
    
    # SH17 class map
    sh17_map = {
        0: "person",
        14: "helmet",
        12: "safety_vest",
        9: "gloves"
    }
    
    # Roboflow class map
    roboflow_map = {
        11: "person",
        3: "helmet", # Hardhat
        13: "safety_vest", # Safety Vest
        1: "gloves" # Gloves
    }
    
    sh17_metrics = extract_metrics(
        r'edge\weights\sh17_yolov8n_best.pt', 
        'sh17', 
        sh17_map, 
        test_frames
    )
    
    rf_metrics = extract_metrics(
        r'runs\detect\edge\weights\training\roboflow\roboflow_yolov8n_640\weights\best.pt', 
        'roboflow', 
        roboflow_map, 
        test_frames
    )
    
    os.makedirs("results/roboflow_vs_sh17", exist_ok=True)
    with open("results/roboflow_vs_sh17/roboflow_vs_sh17.json", "w") as f:
        json.dump({"sh17": sh17_metrics, "roboflow": rf_metrics}, f, indent=4)
        
    md = f"""# Target-Domain Evaluation: SH17 vs Roboflow

Video: JapanPPE.mp4 (50 consecutive frames, start: 287)
Inference: imgsz=640, conf=0.45, GPU

| Metric | SH17 YOLOv8n | Roboflow YOLOv8n |
|---|---:|---:|
| Unique workers | {sh17_metrics['unique_workers']} | {rf_metrics['unique_workers']} |
| Person detections | {sh17_metrics['total_persons']} | {rf_metrics['total_persons']} |
| Helmet detections (Assoc) | {sh17_metrics['total_helmets']} | {rf_metrics['total_helmets']} |
| Safety Vest detections (Assoc) | {sh17_metrics['total_vests']} | {rf_metrics['total_vests']} |
| Gloves (Assoc) | {sh17_metrics['total_gloves']} | {rf_metrics['total_gloves']} |
| Raw Helmet detections | {sh17_metrics.get('raw_helmets', 0)} | {rf_metrics.get('raw_helmets', 0)} |
| Raw Vest detections | {sh17_metrics.get('raw_vests', 0)} | {rf_metrics.get('raw_vests', 0)} |
| FPS | {sh17_metrics['fps']:.1f} | {rf_metrics['fps']:.1f} |
| Latency | {sh17_metrics['latency_ms']:.1f} ms | {rf_metrics['latency_ms']:.1f} ms |
"""
    with open("results/roboflow_vs_sh17/roboflow_vs_sh17.md", "w") as f:
        f.write(md)
        
    print("\n" + md)

if __name__ == "__main__":
    run_benchmark()
