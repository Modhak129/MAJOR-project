import cv2
import time
from ultralytics import YOLO

def run_threshold_test(model_path, video_path, num_frames=50, imgsz=640, conf_thresholds=[0.05, 0.10, 0.20, 0.30, 0.45]):
    print(f"\n==================================================")
    print(f"THRESHOLD TEST: {model_path}")
    print(f"==================================================")
    
    model = YOLO(model_path)
    # Warmup
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        for _ in range(3):
            model(frame, imgsz=imgsz, verbose=False)
    cap.release()
    
    results_map = {}
    PERSON_CLASS_ID = 0
    HELMET_CLASS_ID = 14
    VEST_CLASS_ID = 12
    
    for conf in conf_thresholds:
        cap = cv2.VideoCapture(video_path)
        
        # Fast forward to first person detection
        while True:
            ret, frame = cap.read()
            if not ret: break
            if len(model(frame, verbose=False, imgsz=imgsz, conf=conf)[0].boxes) > 0:
                break
                
        total_time = 0
        frames_processed = 0
        helmet_hits = 0
        vest_hits = 0
        
        while frames_processed < num_frames:
            ret, frame = cap.read()
            if not ret: break
                
            start = time.perf_counter()
            results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, imgsz=imgsz, conf=conf)[0]
            end = time.perf_counter()
            
            persons = []
            items = []
            
            for box in results.boxes:
                cls = int(box.cls[0])
                box_conf = float(box.conf[0])
                worker_id = int(box.id[0]) if box.id is not None else None
                x1, y1, x2, y2 = [float(c) for c in box.xyxy[0]]
                
                obj = {"cls": cls, "id": worker_id, "box": (x1, y1, x2, y2), "conf": box_conf}
                if cls == PERSON_CLASS_ID:
                    persons.append(obj)
                else:
                    items.append(obj)
                    
            if len(persons) == 0:
                continue
                
            total_time += (end - start)
            frames_processed += 1
            
            frame_helmet = False
            frame_vest = False
            
            for p in persons:
                px1, py1, px2, py2 = p["box"]
                p_area = (px2 - px1) * (py2 - py1)
                for item in items:
                    ix1, iy1, ix2, iy2 = item["box"]
                    i_area = (ix2 - ix1) * (iy2 - iy1)
                    
                    inter_x1 = max(px1, ix1)
                    inter_y1 = max(py1, iy1)
                    inter_x2 = min(px2, ix2)
                    inter_y2 = min(py2, iy2)
                    
                    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                        if (inter_area / i_area) > 0.5:
                            if item["cls"] == HELMET_CLASS_ID:
                                frame_helmet = True
                            elif item["cls"] == VEST_CLASS_ID:
                                frame_vest = True
                                
            if frame_helmet: helmet_hits += 1
            if frame_vest: vest_hits += 1
            
        cap.release()
        
        avg_latency = (total_time / frames_processed) * 1000 if frames_processed > 0 else 0
        fps = 1000 / avg_latency if avg_latency > 0 else 0
        helmet_rate = (helmet_hits / frames_processed) * 100 if frames_processed > 0 else 0
        vest_rate = (vest_hits / frames_processed) * 100 if frames_processed > 0 else 0
        
        results_map[conf] = {
            "helmet": helmet_rate,
            "vest": vest_rate,
            "fps": fps,
            "latency": avg_latency
        }
        print(f"Conf {conf}: Helmet {helmet_rate:.1f}%, Vest {vest_rate:.1f}%, FPS {fps:.1f}, Lat {avg_latency:.1f}ms")
        
    return results_map

if __name__ == "__main__":
    import sys
    video_path = r"C:\Major Project\Project-2\data\raw_dataset\source_files\source_files\hardhat.mp4"
    if len(sys.argv) > 1:
        run_threshold_test(sys.argv[1], video_path)
    else:
        run_threshold_test(r"edge\weights\sh17_yolov8n_best.pt", video_path)
        run_threshold_test(r"edge\weights\sh17_yolov8s_best.pt", video_path)
