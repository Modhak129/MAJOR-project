import cv2
import time
import os
import sys
from ultralytics import YOLO

def association_logic(person_box, items_list, helmet_id, vest_id):
    px1, py1, px2, py2 = person_box
    found_items = {"helmet": {"raw": False, "associated": False, "conf": 0.0},
                   "vest": {"raw": False, "associated": False, "conf": 0.0}}
    
    for item in items_list:
        if item["cls"] == helmet_id:
            found_items["helmet"]["raw"] = True
            found_items["helmet"]["conf"] = max(found_items["helmet"]["conf"], item["conf"])
        elif item["cls"] == vest_id:
            found_items["vest"]["raw"] = True
            found_items["vest"]["conf"] = max(found_items["vest"]["conf"], item["conf"])
            
        ix1, iy1, ix2, iy2 = item["box"]
        inter_x1 = max(px1, ix1)
        inter_y1 = max(py1, iy1)
        inter_x2 = min(px2, ix2)
        inter_y2 = min(py2, iy2)
        
        if inter_x2 > inter_x1 and inter_y2 > inter_y1:
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            item_area = (ix2 - ix1) * (iy2 - iy1)
            if inter_area / item_area > 0.5:
                if item["cls"] == helmet_id: found_items["helmet"]["associated"] = True
                elif item["cls"] == vest_id: found_items["vest"]["associated"] = True
                
    return found_items

def run_ab_test(model_path, video_path, num_frames=50, conf_threshold=0.45, imgsz=320):
    print(f"\n==================================================")
    print(f"TESTING MODEL: {model_path}")
    print(f"==================================================")
    model = YOLO(model_path)
    
    names = model.names
    PERSON_CLASS_ID = next((k for k, v in names.items() if v.lower() == "person"), 0)
    HELMET_CLASS_ID = next((k for k, v in names.items() if v.lower() == "helmet"), 14)
    VEST_CLASS_ID = next((k for k, v in names.items() if v.lower() == "safety-vest"), 12)
    
    cap = cv2.VideoCapture(video_path)
    
    # Warmup
    ret, frame = cap.read()
    if ret:
        for _ in range(3):
            model(frame, imgsz=imgsz, verbose=False)
    
    total_time = 0
    frames_processed = 0
    helmet_hits = 0
    vest_hits = 0
    
    # Fast forward to first person detection
    while True:
        ret, frame = cap.read()
        if not ret: break
        if len(model(frame, verbose=False, imgsz=imgsz, conf=conf_threshold)[0].boxes) > 0:
            break

    while frames_processed < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        start = time.perf_counter()
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, imgsz=imgsz, conf=conf_threshold)[0]
        end = time.perf_counter()
        
        persons = []
        items = []
        
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            worker_id = int(box.id[0]) if box.id is not None else None
            x1, y1, x2, y2 = [float(c) for c in box.xyxy[0]]
            
            obj = {"cls": cls, "id": worker_id, "box": (x1, y1, x2, y2), "conf": conf}
            if cls == PERSON_CLASS_ID:
                persons.append(obj)
            else:
                items.append(obj)
        
        if len(persons) == 0:
            continue # Skip frames without people to ensure we test exactly 50 person-positive frames
            
        total_time += (end - start)
        frames_processed += 1
                
        # Debugging Output
        print(f"\nFrame {frames_processed}:")
        frame_helmet_raw = False
        frame_vest_raw = False
        
        for p in persons:
            p_state = association_logic(p["box"], items, HELMET_CLASS_ID, VEST_CLASS_ID)
            wid = p["id"] if p["id"] is not None else "UNTRACKED"
            print(f"Worker #{wid}")
            print(f"  Helmet: raw detected = {p_state['helmet']['raw']}, associated = {p_state['helmet']['associated']}, confidence = {p_state['helmet']['conf']:.2f}")
            print(f"  Safety-vest: raw detected = {p_state['vest']['raw']}, associated = {p_state['vest']['associated']}, confidence = {p_state['vest']['conf']:.2f}")
            
            if p_state['helmet']['associated']: frame_helmet_raw = True
            if p_state['vest']['associated']: frame_vest_raw = True
            
        if frame_helmet_raw: helmet_hits += 1
        if frame_vest_raw: vest_hits += 1

    cap.release()
    
    avg_latency = (total_time / frames_processed) * 1000 if frames_processed > 0 else 0
    fps = frames_processed / total_time if total_time > 0 else 0
    helmet_rate = (helmet_hits / frames_processed) * 100 if frames_processed > 0 else 0
    vest_rate = (vest_hits / frames_processed) * 100 if frames_processed > 0 else 0
    
    return {
        "fps": fps,
        "latency": avg_latency,
        "helmet_rate": helmet_rate,
        "vest_rate": vest_rate
    }

def main():
    video_path = "C:/Major Project/Project-2/data/raw_dataset/source_files/source_files/JapanPPE.mp4"
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return

    res_n = run_ab_test("edge/weights/sh17_yolov8n_best.pt", video_path, num_frames=50, imgsz=640)
    res_s = run_ab_test("edge/weights/sh17_yolov8s_best.pt", video_path, num_frames=50, imgsz=640)

    print("\n==================================================")
    print("FINAL COMPARISON RESULTS")
    print("==================================================")
    
    print("\nYOLOv8n")
    print(f"Helmet detection rate: {res_n['helmet_rate']:.1f}%")
    print(f"Safety-vest detection rate: {res_n['vest_rate']:.1f}%")
    print(f"FPS: {res_n['fps']:.1f}")
    print(f"Latency: {res_n['latency']:.1f} ms")
    
    print("\nYOLOv8s")
    print(f"Helmet detection rate: {res_s['helmet_rate']:.1f}%")
    print(f"Safety-vest detection rate: {res_s['vest_rate']:.1f}%")
    print(f"FPS: {res_s['fps']:.1f}")
    print(f"Latency: {res_s['latency']:.1f} ms")

if __name__ == "__main__":
    main()
