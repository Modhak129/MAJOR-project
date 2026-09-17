import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import sys
import cv2
import numpy as np
import threading
import torch
import os

# Prevent PyTorch from spawning massive threads leading to CPU thrashing
torch.set_num_threads(2)
os.environ["OMP_NUM_THREADS"] = "2"

from ultralytics import YOLO
from alerts import AlertManager
from xai_stub import generate_saliency_map

class InferenceEngine:
    def __init__(self, config, model_path, client_id, dashboard_url):
        self.config = config
        self.model_lock = threading.Lock()
        self.model = YOLO(model_path)
        self.client_id = client_id
        self.min_conf = config['inference']['min_confidence']
        self.alert_manager = AlertManager(dashboard_url, config['inference']['alert_cooldown_sec'])
        self.geofence_polygon = []
        
    def reload_model(self, weights_path):
        with self.model_lock:
            self.model = YOLO(weights_path)
            print(f"[{self.client_id}] Hot-swapped new global model weights.")
        
    def update_geofence(self, polygon):
        self.geofence_polygon = polygon

    def process_frame(self, frame):
        import os
        import time
        from shapely.geometry import Polygon, box as shapely_box
        
        start_t = time.time()
        violation_found = False
        labels = []
        
        with self.model_lock:
            # Optimize inference speed by reducing input resolution to guarantee high FPS on CPU and SocketIO
            imgsz_val = 320
            device = '0' if sys.platform == 'win32' else 'cpu'
    
            t0 = time.time()
            results = self.model.track(frame, persist=True,   tracker="bytetrack.yaml", verbose=False, imgsz=imgsz_val, device=device)[0]

            t1 = time.time()
            print(f"[{self.client_id}] YOLO track took {(t1-t0)*1000:.1f}ms", flush=True)
        
        persons = []
        items = []
        
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < self.min_conf:
                continue
            cls = int(box.cls[0])
            worker_id = int(box.id[0]) if box.id is not None else None
            
            x_c, y_c, w, h = float(box.xywhn[0][0]), float(box.xywhn[0][1]), float(box.xywhn[0][2]), float(box.xywhn[0][3])
            labels.append(f"{cls} {x_c} {y_c} {w} {h}")
            
            x1, y1, x2, y2 = float(box.xyxy[0][0]), float(box.xyxy[0][1]), float(box.xyxy[0][2]), float(box.xyxy[0][3])
            obj = {"cls": cls, "id": worker_id, "box": (x1, y1, x2, y2), "conf": conf}
            if cls == 5:
                persons.append(obj)
            else:
                items.append(obj)

        geo_poly = Polygon(self.geofence_polygon) if len(self.geofence_polygon) >= 3 else None

        print(f"[{self.client_id}] Detections -> Persons: {len(persons)}, Items: {[i['cls'] for i in items]}", flush=True)

        for person in persons:
            px1, py1, px2, py2 = person["box"]
            p_area = (px2 - px1) * (py2 - py1)
            
            found_items = set()
            for item in items:
                ix1, iy1, ix2, iy2 = item["box"]
                # Compute intersection
                inter_x1 = max(px1, ix1)
                inter_y1 = max(py1, iy1)
                inter_x2 = min(px2, ix2)
                inter_y2 = min(py2, iy2)
                
                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    item_area = (ix2 - ix1) * (iy2 - iy1)
                    if inter_area / item_area > 0.5: # 50% contained in person box
                        if item["cls"] == 0: found_items.add("helmet")
                        elif item["cls"] == 7: found_items.add("jacket")
            
            missing = []
            if "helmet" not in found_items: missing.append("helmet")
            if "jacket" not in found_items: missing.append("jacket")
            
            print(f"[{self.client_id}] Person box: {person['box']} | Found items: {found_items} | Missing: {missing}", flush=True)
            
            if missing:
                violation_found = True
                details = {"missing_items": missing}
                self.alert_manager.send_alert(self.client_id, "Missing PPE", person["conf"], worker_id=person["id"], details=details)
                
            # Geofencing logic (use person box)
            if geo_poly:
                person_box = shapely_box(px1, py1, px2, py2)
                if person_box.intersects(geo_poly):
                    violation_found = True
                    details = "Worker entered restricted zone"
                    self.alert_manager.send_alert(self.client_id, "Zone Intrusion", person["conf"], worker_id=person["id"], details=details)

        if violation_found:
            os.makedirs("/app/data/local_training/images", exist_ok=True)
            os.makedirs("/app/data/local_training/labels", exist_ok=True)
            timestamp = int(time.time() * 1000)
            img_path = f"/app/data/local_training/images/frame_{timestamp}.jpg"
            lbl_path = f"/app/data/local_training/labels/frame_{timestamp}.txt"
            cv2.imwrite(img_path, frame)
            with open(lbl_path, "w") as f:
                f.write("\n".join(labels))
                
        plot_frame = results.plot()
        if len(self.geofence_polygon) >= 3:
            pts = np.array(self.geofence_polygon, np.int32).reshape((-1, 1, 2))
            cv2.polylines(plot_frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

        frame_with_xai = generate_saliency_map(plot_frame, self.model)
        return frame_with_xai
