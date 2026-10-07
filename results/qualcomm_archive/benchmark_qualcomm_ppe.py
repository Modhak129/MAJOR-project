import cv2
import time
import os
import sys
import json
from ultralytics import YOLO

sys.path.append(os.path.abspath('edge'))
from qualcomm_ppe_detector import QualcommPPEDetector
from inference import expand_bbox_for_ppe

def run_benchmark():
    video_path = os.path.abspath(r'data\raw_dataset\source_files\source_files\JapanPPE.mp4')
    yolo_model_path = os.path.abspath(r'edge\weights\sh17_yolov8n_best.pt')
    q_model_path = os.path.abspath(r'models\qualcomm\gear_guard_net-onnx-float\gear_guard_net.onnx')
    
    print("Loading models...")
    yolo_model = YOLO(yolo_model_path)
    q_detector = QualcommPPEDetector(q_model_path, conf_thresh=0.45)
    
    cap = cv2.VideoCapture(video_path)
    
    # Fast forward to first person detection with YOLO to align with previous tests
    frame_idx = 0
    start_frame = -1
    print("Fast-forwarding to first person detection...")
    while True:
        ret, frame = cap.read()
        if not ret: break
        res = yolo_model(frame, verbose=False, imgsz=320, conf=0.45)[0]
        person_detected = False
        for box in res.boxes:
            if int(box.cls[0]) == 0:
                person_detected = True
                break
        if person_detected:
            start_frame = frame_idx
            break
        frame_idx += 1
        
    print(f"First person detected at frame {start_frame}")
    
    test_frames = []
    test_frames.append(frame)
    while len(test_frames) < 50:
        ret, frame = cap.read()
        if not ret: break
        test_frames.append(frame)
    cap.release()
    print(f"Captured {len(test_frames)} frames for benchmark.")
    
    expansions = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    results_list = []
    
    for exp in expansions:
        print(f"\nRunning Expansion Sweep: {exp*100:.0f}%")
        # Reset tracker for each sweep
        yolo_model.predictor = None
        
        metrics = {
            "expansion": exp,
            "total_persons": 0,
            "unique_workers": set(),
            "helmets_found": 0,
            "vests_found": 0,
            "helmet_conf_sum": 0.0,
            "vest_conf_sum": 0.0,
            "pipeline_latency_ms": 0.0,
            "ppe_latency_ms": 0.0
        }
        
        t0 = time.time()
        q_latency_only = 0
        q_persons = 0
        
        for frame in test_frames:
            res = yolo_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, imgsz=320, device='cpu')[0]
            persons = []
            
            for box in res.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                wid = int(box.id[0]) if box.id is not None else -1
                if conf < 0.45: continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                if cls == 0: persons.append({'box': [x1, y1, x2, y2], 'id': wid})
                
            metrics["total_persons"] += len(persons)
            
            for p in persons:
                px1, py1, px2, py2 = p['box']
                metrics["unique_workers"].add(p['id'])
                
                frame_h, frame_w = frame.shape[:2]
                cx1, cy1, cx2, cy2 = expand_bbox_for_ppe(px1, py1, px2, py2, frame_w, frame_h, exp)
                crop = frame[cy1:cy2, cx1:cx2]
                
                if crop.size == 0: continue
                
                t_s = time.time()
                q_res = q_detector.detect(crop)
                t_e = time.time()
                
                q_latency_only += (t_e - t_s)
                q_persons += 1
                
                found_helmet = False
                found_vest = False
                for ppe in q_res:
                    if ppe['class_name'] == 'helmet':
                        found_helmet = True
                        metrics["helmet_conf_sum"] += ppe['confidence']
                    if ppe['class_name'] == 'safety_vest':
                        found_vest = True
                        metrics["vest_conf_sum"] += ppe['confidence']
                
                if found_helmet: metrics["helmets_found"] += 1
                if found_vest: metrics["vests_found"] += 1
                
        t1 = time.time()
        metrics["pipeline_latency_ms"] = (t1 - t0) * 1000 / len(test_frames)
        metrics["ppe_latency_ms"] = (q_latency_only * 1000 / q_persons) if q_persons > 0 else 0
        metrics["unique_workers"] = len(metrics["unique_workers"])
        
        results_list.append(metrics)
        
    os.makedirs('results', exist_ok=True)
    with open('results/qualcomm_crop_expansion_benchmark.json', 'w') as f:
        json.dump(results_list, f, indent=4)
        
    report = f"""# Qualcomm PPE Crop Expansion Benchmark

Model: Qualcomm PPE-Detection (gear_guard_net-onnx-float)
Input: JapanPPE.mp4 (50 consecutive frames starting at {start_frame})

| Expansion | Unique Workers | Person Crops | Helmet Detections | Vest Detections | Helmet Rate | Vest Rate | Avg H-Conf | Avg V-Conf | PPE ms | Pipeline ms | FPS |
|-----------|----------------|--------------|-------------------|-----------------|-------------|-----------|------------|------------|--------|--------------|-----|
"""

    for r in results_list:
        h_rate = r['helmets_found'] / max(1, r['total_persons']) * 100
        v_rate = r['vests_found'] / max(1, r['total_persons']) * 100
        avg_h_conf = r['helmet_conf_sum'] / max(1, r['helmets_found'])
        avg_v_conf = r['vest_conf_sum'] / max(1, r['vests_found'])
        fps = 1000 / max(0.1, r['pipeline_latency_ms'])
        
        report += f"| {r['expansion']*100:.0f}% | {r['unique_workers']} | {r['total_persons']} | {r['helmets_found']} | {r['vests_found']} | {h_rate:.1f}% | {v_rate:.1f}% | {avg_h_conf:.2f} | {avg_v_conf:.2f} | {r['ppe_latency_ms']:.1f} | {r['pipeline_latency_ms']:.1f} | {fps:.1f} |\n"

    with open('results/qualcomm_crop_expansion_benchmark.md', 'w') as f:
        f.write(report)
        
    print("\n" + report)

if __name__ == "__main__":
    run_benchmark()
