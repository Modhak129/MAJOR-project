import cv2
import time
import sys
import os
import json
import numpy as np
sys.path.append(os.path.abspath('edge'))
from ultralytics import YOLO

def benchmark_video(model, model_name, video_name, video_path, frames_to_process, imgsz=640, conf=0.45):
    print(f"\n--- Benchmarking {model_name} on {video_name} ---")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open {video_path}")
        return None
        
    metrics = {
        "model": model_name,
        "video": video_name,
        "total_persons": 0,
        "total_helmets": 0,
        "total_vests": 0,
        "total_gloves": 0,
        "unique_workers": set(),
        "latency_ms": 0.0,
        "fps": 0.0,
        "confidences": {
            "helmet": [],
            "vest": [],
            "gloves": []
        }
    }
    
    sh17_map = {0: "person", 14: "helmet", 12: "safety_vest", 9: "gloves"}
    
    t0_total = time.time()
    
    vis_dir = os.path.join("results", "multi_video_comparison", video_name.split('.')[0])
    os.makedirs(vis_dir, exist_ok=True)
    
    frames_processed = 0
    cap_frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Check if this frame should be processed
        if frames_to_process is not None and cap_frame_idx not in frames_to_process:
            cap_frame_idx += 1
            continue
            
        cap_frame_idx += 1
        frames_processed += 1
        
        res = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, imgsz=imgsz, device=0, conf=conf)[0]
        
        # Visualization
        if frames_processed == 50 or frames_processed == min(100, len(frames_to_process) if frames_to_process else 100):
            annotated = res.plot()
            cv2.imwrite(os.path.join(vis_dir, f"frame_{cap_frame_idx:06d}_{model_name}.jpg"), annotated)
            
        persons = []
        items = []
        
        for box in res.boxes:
            cls_id = int(box.cls[0])
            c_val = float(box.conf[0])
            wid = int(box.id[0]) if box.id is not None else -1
            if c_val < conf: continue
            
            if cls_id not in sh17_map: continue
            semantic_cls = sh17_map[cls_id]
            
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            if semantic_cls == "person":
                persons.append({'box': [x1, y1, x2, y2], 'id': wid})
            else:
                items.append({'cls': semantic_cls, 'box': [x1, y1, x2, y2], 'conf': c_val})
                
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
                        if item['cls'] == 'helmet': metrics['confidences']['helmet'].append(item['conf'])
                        elif item['cls'] == 'safety_vest': metrics['confidences']['vest'].append(item['conf'])
                        elif item['cls'] == 'gloves': metrics['confidences']['gloves'].append(item['conf'])
                        
            if 'helmet' in found_items: metrics["total_helmets"] += 1
            if 'safety_vest' in found_items: metrics["total_vests"] += 1
            if 'gloves' in found_items: metrics["total_gloves"] += 1
            
        if frames_to_process is not None and frames_processed >= len(frames_to_process):
            break

    cap.release()
    t1_total = time.time()
    
    metrics["latency_ms"] = (t1_total - t0_total) * 1000 / frames_processed
    metrics["fps"] = 1000 / metrics["latency_ms"]
    metrics["unique_workers"] = len(metrics["unique_workers"])
    
    # Calculate confidence stats
    for k in metrics["confidences"]:
        vals = metrics["confidences"][k]
        if vals:
            metrics[f"{k}_mean_conf"] = float(np.mean(vals))
            metrics[f"{k}_median_conf"] = float(np.median(vals))
            metrics[f"{k}_max_conf"] = float(np.max(vals))
        else:
            metrics[f"{k}_mean_conf"] = 0.0
            metrics[f"{k}_median_conf"] = 0.0
            metrics[f"{k}_max_conf"] = 0.0
            
    # Clean up large arrays
    del metrics["confidences"]
    
    print(f"Processed {frames_processed} frames. FPS: {metrics['fps']:.1f}")
    return metrics

def build_frames_set(total_frames, strategy):
    if strategy == "all":
        return set(range(total_frames))
    elif strategy == "japanppe_50":
        return set(range(287, 287+50))
    elif strategy == "3x300":
        # first 300, middle 300, last 300
        if total_frames <= 900: return set(range(total_frames))
        f = set(range(0, 300))
        mid = total_frames // 2
        f.update(range(mid - 150, mid + 150))
        f.update(range(total_frames - 300, total_frames))
        return f
    return set(range(total_frames))

def run_multi_benchmark():
    os.makedirs("results/multi_video_comparison", exist_ok=True)
    
    videos = [
        {"name": "JapanPPE.mp4", "path": r"data\raw_dataset\source_files\source_files\JapanPPE.mp4", "strategy": "japanppe_50", "total": 4784},
        {"name": "JapanPPE.mp4", "path": r"data\raw_dataset\source_files\source_files\JapanPPE.mp4", "strategy": "3x300", "total": 4784},
        {"name": "hardhat.mp4", "path": r"data\raw_dataset\source_files\source_files\hardhat.mp4", "strategy": "all", "total": 1826},
        {"name": "indianworkers.mp4", "path": r"data\raw_dataset\source_files\source_files\indianworkers.mp4", "strategy": "all", "total": 648}
    ]
    
    models = {
        "SH17_v8n": r"edge\weights\sh17_yolov8n_best.pt",
        "SH17_v8s": r"edge\weights\sh17_yolov8s_best.pt"
    }
    
    loaded_models = {}
    for name, path in models.items():
        if os.path.exists(path):
            print(f"Loading {name}...")
            loaded_models[name] = YOLO(path)
        else:
            print(f"WARNING: {name} not found at {path}")
            
    all_results = []
    
    for v in videos:
        v_name = v["name"]
        if v["strategy"] == "japanppe_50":
            v_label = "JapanPPE_50frames"
        elif v["strategy"] == "3x300":
            v_label = "JapanPPE_900frames"
        else:
            v_label = v_name.split(".")[0]
            
        frames_to_process = build_frames_set(v["total"], v["strategy"])
        
        v_res = {}
        for m_name, model in loaded_models.items():
            res = benchmark_video(model, m_name, v_name, v["path"], frames_to_process)
            v_res[m_name] = res
            if res:
                all_results.append({"video": v_label, "metrics": res})
                
        with open(f"results/multi_video_comparison/{v_label}_results.json", "w") as f:
            json.dump(v_res, f, indent=4)
            
    with open("results/multi_video_comparison/multi_video_results.json", "w") as f:
        json.dump(all_results, f, indent=4)
        
    print("Done!")
    
if __name__ == "__main__":
    run_multi_benchmark()
