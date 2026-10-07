import os
import cv2
import time
import json
import numpy as np
from ultralytics import YOLO

def main():
    stage1_path = r"C:\Major Project\Project-2\edge\weights\clean_sh17_yolov8n_640_best.pt"
    stage2_path = r"C:\Major Project\Project-2\runs\detect\stage2_vest_final_20pct\weights\best.pt"
    
    videos = [
        r"C:\Major Project\Project-2\data\raw_dataset\source_files\source_files\JapanPPE.mp4",
        r"C:\Major Project\Project-2\data\raw_dataset\source_files\source_files\hardhat.mp4",
        r"C:\Major Project\Project-2\data\raw_dataset\source_files\source_files\indianworkers.mp4"
    ]
    
    out_dir = r"C:\Major Project\Project-2\runs\audits\stage2_vest_target_benchmark"
    vis_dir = os.path.join(out_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    
    stage1_model = YOLO(stage1_path)
    stage2_model = YOLO(stage2_path)
    
    person_cls_sh17 = None
    for k, v in stage1_model.names.items():
        if v.lower() == 'person':
            person_cls_sh17 = k
            break
            
    if person_cls_sh17 is None:
        person_cls_sh17 = 5 # default for SH17
        
    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    stage1_conf = 0.45
    expansion = 0.20
    
    results = {}
    
    for vid_path in videos:
        vid_name = os.path.basename(vid_path)
        print(f"\nProcessing {vid_name}...")
        
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            print(f"Could not open {vid_name}")
            continue
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps else 0
        
        print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}, Duration: {duration:.2f}s")
        
        # We will process a subset of frames if the video is too long, or just process all frames.
        # To be safe and fast, let's process 1 frame every 5 frames.
        frame_stride = max(1, int(fps / 5)) if fps > 0 else 5
        
        vid_results = {}
        for th in thresholds:
            vid_results[th] = {
                "total_frames": total_frames,
                "sampled_frames": 0,
                "frames_with_person": 0,
                "person_detections": 0,
                "vest_detections": 0,
                "frames_with_vest": 0,
                "vest_confs": [],
                "latencies": []
            }
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue
                
            start_t = time.time()
            
            # Stage 1
            st1_res = stage1_model.predict(frame, imgsz=640, conf=stage1_conf, verbose=False, classes=[person_cls_sh17])
            persons = []
            if len(st1_res) > 0 and len(st1_res[0].boxes) > 0:
                for box in st1_res[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    persons.append([int(x1), int(y1), int(x2), int(y2), conf])
                    
            # Stage 2 for each threshold
            # To optimize, we can run Stage 2 once per crop and evaluate against thresholds
            crop_results = []
            
            for p_idx, (px1, py1, px2, py2, pconf) in enumerate(persons):
                pw = px2 - px1
                ph = py2 - py1
                exp_w = pw * expansion
                exp_h = ph * expansion
                nx1 = max(0, int(px1 - exp_w/2))
                ny1 = max(0, int(py1 - exp_h/2))
                nx2 = min(width, int(px2 + exp_w/2))
                ny2 = min(height, int(py2 + exp_h/2))
                
                if nx2 - nx1 > 10 and ny2 - ny1 > 10:
                    crop = frame[ny1:ny2, nx1:nx2]
                    st2_res = stage2_model.predict(crop, imgsz=640, verbose=False)
                    best_vest_conf = 0.0
                    if len(st2_res) > 0 and len(st2_res[0].boxes) > 0:
                        for b2 in st2_res[0].boxes:
                            if int(b2.cls[0]) == 0:
                                best_vest_conf = max(best_vest_conf, float(b2.conf[0]))
                                
                    crop_results.append(best_vest_conf)
                    
            end_t = time.time()
            latency = (end_t - start_t) * 1000
            
            # Accumulate metrics
            for th in thresholds:
                vr = vid_results[th]
                vr["sampled_frames"] += 1
                vr["latencies"].append(latency)
                
                if len(persons) > 0:
                    vr["frames_with_person"] += 1
                    vr["person_detections"] += len(persons)
                    
                vests_found = sum(1 for conf in crop_results if conf >= th)
                if vests_found > 0:
                    vr["frames_with_vest"] += 1
                    vr["vest_detections"] += vests_found
                    vr["vest_confs"].extend([conf for conf in crop_results if conf >= th])
                    
            # Visual validation (save a few frames per video for a specific threshold like 0.35)
            if len(persons) > 0 and len(os.listdir(vis_dir)) < 150: # save up to 50 images total
                vis_frame = frame.copy()
                has_vest = False
                for p_idx, (px1, py1, px2, py2, pconf) in enumerate(persons):
                    cv2.rectangle(vis_frame, (px1, py1), (px2, py2), (255, 0, 0), 2)
                    v_conf = crop_results[p_idx]
                    if v_conf >= 0.35:
                        has_vest = True
                        cv2.putText(vis_frame, f"Vest: {v_conf:.2f}", (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        cv2.putText(vis_frame, f"No Vest", (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                if frame_idx % (frame_stride * 5) == 0 or has_vest:
                    cv2.imwrite(os.path.join(vis_dir, f"{vid_name}_f{frame_idx}.jpg"), vis_frame)

            frame_idx += 1
            
        cap.release()
        
        # Aggregate
        for th in thresholds:
            vr = vid_results[th]
            vr["avg_latency_ms"] = np.mean(vr["latencies"]) if vr["latencies"] else 0
            vr["fps_effective"] = 1000 / vr["avg_latency_ms"] if vr["avg_latency_ms"] > 0 else 0
            vr["avg_vest_conf"] = np.mean(vr["vest_confs"]) if vr["vest_confs"] else 0
            vr["max_vest_conf"] = np.max(vr["vest_confs"]) if vr["vest_confs"] else 0
            vr["min_vest_conf"] = np.min(vr["vest_confs"]) if vr["vest_confs"] else 0
            
            # clean up raw lists
            del vr["latencies"]
            del vr["vest_confs"]
            
        results[vid_name] = vid_results
        
    with open(os.path.join(out_dir, "benchmark_results.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print("Benchmark completed.")

if __name__ == "__main__":
    main()
