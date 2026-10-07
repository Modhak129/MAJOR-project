import cv2
import time
import sys
import os
sys.path.append(os.path.abspath('edge'))
from ultralytics import YOLO

def run_test():
    video_path = os.path.abspath(r'data\raw_dataset\source_files\source_files\JapanPPE.mp4')
    yolo_model_path = os.path.abspath(r'edge\weights\sh17_yolov8n_best.pt')
    
    print("Loading SH17 model...")
    yolo_model = YOLO(yolo_model_path)
    cap = cv2.VideoCapture(video_path)
    
    frame_idx = 0
    start_frame = -1
    print("Fast-forwarding to frame 287...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 287)
    
    test_frames = []
    while len(test_frames) < 50:
        ret, frame = cap.read()
        if not ret: break
        test_frames.append(frame)
    cap.release()
    print(f"Captured {len(test_frames)} frames.")
    
    t0 = time.time()
    
    total_persons = 0
    total_helmets = 0
    total_vests = 0
    total_gloves = 0
    unique_workers = set()
    
    for frame in test_frames:
        res = yolo_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, imgsz=320, device=0)[0]
        
        persons = []
        items = []
        for box in res.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            wid = int(box.id[0]) if box.id is not None else -1
            if conf < 0.45: continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            if cls == 0:
                persons.append({'box': [x1, y1, x2, y2], 'id': wid})
            elif cls == 14:
                items.append({'cls': 'helmet', 'box': [x1, y1, x2, y2]})
            elif cls == 12:
                items.append({'cls': 'vest', 'box': [x1, y1, x2, y2]})
            elif cls == 9:
                items.append({'cls': 'gloves', 'box': [x1, y1, x2, y2]})
                
        total_persons += len(persons)
        
        for person in persons:
            px1, py1, px2, py2 = person['box']
            unique_workers.add(person['id'])
            
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
                        
            if 'helmet' in found_items: total_helmets += 1
            if 'vest' in found_items: total_vests += 1
            if 'gloves' in found_items: total_gloves += 1

    t1 = time.time()
    latency_ms = (t1 - t0) * 1000 / len(test_frames)
    fps = 1000 / latency_ms
    
    print("\n--- SH17 JAPAN PPE RESULTS ---")
    print(f"Frames processed: {len(test_frames)}")
    print(f"Unique workers: {len(unique_workers)}")
    print(f"Person detections: {total_persons}")
    print(f"Helmet detections: {total_helmets}")
    print(f"Safety vest detections: {total_vests}")
    print(f"Gloves detections: {total_gloves}")
    print(f"Latency: {latency_ms:.1f} ms/frame")
    print(f"FPS: {fps:.1f}")

if __name__ == "__main__":
    run_test()
