import cv2
import time
import sys
import os
import json
import numpy as np

sys.path.append(os.path.abspath('edge'))
from ultralytics import YOLO

def expand_crop(x1, y1, x2, y2, H, W, expansion_ratio):
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    new_w = w * (1 + expansion_ratio)
    new_h = h * (1 + expansion_ratio)
    
    nx1 = max(0, int(cx - new_w / 2))
    ny1 = max(0, int(cy - new_h / 2))
    nx2 = min(W, int(cx + new_w / 2))
    ny2 = min(H, int(cy + new_h / 2))
    return nx1, ny1, nx2, ny2

def run_experiment():
    video_path = os.path.abspath(r'data\raw_dataset\source_files\source_files\JapanPPE.mp4')
    sh17_model_path = os.path.abspath(r'edge\weights\sh17_yolov8n_best.pt')
    rf_model_path = os.path.abspath(r'runs\detect\edge\weights\training\roboflow\roboflow_yolov8n_640\weights\best.pt')
    
    print("Loading models...")
    sh17 = YOLO(sh17_model_path)
    rf = YOLO(rf_model_path)
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 287)
    test_frames = []
    while len(test_frames) < 50:
        ret, frame = cap.read()
        if not ret: break
        test_frames.append(frame)
    cap.release()
    print(f"Captured {len(test_frames)} frames.")
    
    rf_map = {3: "helmet", 13: "safety_vest", 1: "gloves"}
    
    expansions = [0.0, 0.05, 0.10, 0.15, 0.20]
    thresholds = [0.05, 0.10, 0.20, 0.30, 0.40, 0.45, 0.50, 0.60, 0.70]
    
    results = {}
    
    os.makedirs("results/three_model_comparison/visuals", exist_ok=True)
    
    for exp in expansions:
        results[exp] = {}
        for conf in thresholds:
            metrics = {
                "total_persons": 0,
                "total_helmets": 0,
                "total_vests": 0,
                "total_gloves": 0,
                "unique_workers": set(),
                "sh17_latency": 0.0,
                "rf_latency": 0.0,
                "total_latency": 0.0
            }
            
            t_total_start = time.time()
            t_sh17 = 0
            t_rf = 0
            
            frame_idx = 0
            for frame in test_frames:
                frame_idx += 1
                H, W = frame.shape[:2]
                
                vis_frame = frame.copy() if exp == 0.20 and conf == 0.45 and (frame_idx % 10 == 0 or frame_idx == 1) else None
                
                t0 = time.time()
                res_sh17 = sh17.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, imgsz=640, device=0, conf=0.45)[0]
                t_sh17 += time.time() - t0
                
                persons = []
                for box in res_sh17.boxes:
                    c = int(box.cls[0])
                    cc = float(box.conf[0])
                    if cc < 0.45 or c != 0: continue
                    wid = int(box.id[0]) if box.id is not None else -1
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    persons.append({'id': wid, 'box': (x1, y1, x2, y2)})
                    
                    if vis_frame is not None:
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(vis_frame, f"Person {wid}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        
                metrics["total_persons"] += len(persons)
                
                for p in persons:
                    metrics["unique_workers"].add(p['id'])
                    x1, y1, x2, y2 = p['box']
                    nx1, ny1, nx2, ny2 = expand_crop(x1, y1, x2, y2, H, W, exp)
                    
                    crop = frame[ny1:ny2, nx1:nx2]
                    if crop.size == 0: continue
                    
                    t0 = time.time()
                    res_rf = rf(crop, imgsz=320, device=0, conf=conf, verbose=False)[0]
                    t_rf += time.time() - t0
                    
                    found_items = set()
                    for box in res_rf.boxes:
                        c = int(box.cls[0])
                        if c in rf_map:
                            found_items.add(rf_map[c])
                            if vis_frame is not None:
                                bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
                                # Translate back to full frame
                                bx1 += nx1
                                bx2 += nx1
                                by1 += ny1
                                by2 += ny1
                                cv2.rectangle(vis_frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                                cv2.putText(vis_frame, rf_map[c], (bx1, by1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                
                    if 'helmet' in found_items: metrics['total_helmets'] += 1
                    if 'safety_vest' in found_items: metrics['total_vests'] += 1
                    if 'gloves' in found_items: metrics['total_gloves'] += 1
                    
                if vis_frame is not None:
                    cv2.imwrite(f"results/three_model_comparison/visuals/frame_{287+frame_idx-1:04d}_twostage.jpg", vis_frame)
                    
            t_total_end = time.time()
            metrics['unique_workers'] = len(metrics['unique_workers'])
            metrics['sh17_latency'] = (t_sh17 * 1000) / len(test_frames)
            # Latency per crop for rf
            metrics['rf_latency'] = (t_rf * 1000) / max(1, metrics['total_persons'])
            metrics['total_latency'] = ((t_total_end - t_total_start) * 1000) / len(test_frames)
            metrics['fps'] = 1000 / metrics['total_latency']
            
            results[exp][conf] = metrics
            print(f"Exp: {exp:.2f}, Conf: {conf:.2f} | Helmets: {metrics['total_helmets']}, Vests: {metrics['total_vests']}, Gloves: {metrics['total_gloves']}")
            
    with open("results/three_model_comparison/three_model_comparison.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("Benchmark complete!")

if __name__ == "__main__":
    run_experiment()
