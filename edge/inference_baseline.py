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
        self.vest_conf = config['inference'].get('vest_confidence', 0.35)
        self.alert_manager = AlertManager(dashboard_url, config['inference']['alert_cooldown_sec'])
        self.geofence_polygon = []
        
        self.ppe_detector_mode = config['inference'].get('ppe_detector_mode', 'sh17')
        self.helmet_conf = config['inference'].get('helmet_conf', 0.40)
        self.helmet_crop_expansion = config['inference'].get('helmet_crop_expansion', 0.20)
        
        self.vest_crop_expansion = config['inference'].get('vest_crop_expansion', 0.20)
        
        print(f"[{self.client_id}] InferenceEngine initialized. vest_confidence = {self.vest_conf}", flush=True)
        
        self.helmet_model = None
        self.vest_model = None
        
        if self.ppe_detector_mode == "two_stage":
            helmet_model_path = config['inference'].get('helmet_model_path')
            if helmet_model_path and os.path.exists(helmet_model_path):
                self.helmet_model = YOLO(helmet_model_path)
                print(f"[{self.client_id}] Loaded Stage-2 Helmet detector: {helmet_model_path}")
            else:
                print(f"[{self.client_id}] Warning: two_stage mode enabled but helmet model path is invalid!")
                
            vest_model_path = config['inference'].get('vest_model_path')
            if vest_model_path and os.path.exists(vest_model_path):
                self.vest_model = YOLO(vest_model_path)
                vest_cls_name = self.vest_model.names.get(0, 'safety_vest')
                print(f"[{self.client_id}] Loaded Stage-2 Vest detector: {vest_model_path}")
                print(f"[{self.client_id}] Stage-2 Vest class mapping: 0 = {vest_cls_name}")
                print(f"[{self.client_id}] Stage-2 Vest Confidence: {self.vest_conf}, Crop Expansion: {self.vest_crop_expansion}")
            else:
                print(f"[{self.client_id}] Warning: two_stage mode enabled but vest model path is invalid!")

    def reload_model(self, weights_path):
        with self.model_lock:
            self.model = YOLO(weights_path)
            print(f"[{self.client_id}] Hot-swapped new global model weights.")
        
    def update_geofence(self, polygon):
        self.geofence_polygon = polygon

    def reset_tracker(self):
        with self.model_lock:
            if hasattr(self.model, 'predictor') and self.model.predictor is not None:
                self.model.predictor = None
            if hasattr(self, 'alert_manager'):
                self.alert_manager.history = {}
            print(f"[{self.client_id}] Tracker state and Worker IDs have been reset for new video.")

    def process_frame(self, frame):
        import time
        from shapely.geometry import Polygon, box as shapely_box
        
        start_t = time.time()
        violation_found = False
        labels = []
        
        # Avoid running full detection if frame is invalid
        if frame is None or frame.size == 0:
            return frame
            
        H, W = frame.shape[:2]
        
        with self.model_lock:
            imgsz_val = self.config['inference'].get('input_resolution', 320)
            device = '0' if sys.platform == 'win32' else 'cpu'
            results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, imgsz=imgsz_val, device=device)[0]
        
        persons = []
        items = []
        
        names = self.model.names
        PERSON_CLASS_ID = next((k for k, v in names.items() if v.lower() == "person"), 0)
        HELMET_CLASS_ID = next((k for k, v in names.items() if v.lower() == "helmet"), 14)
        VEST_CLASS_ID = next((k for k, v in names.items() if v.lower() == "safety-vest"), 12)
        GLOVES_CLASS_ID = next((k for k, v in names.items() if v.lower() == "gloves"), 9)

        # In two_stage mode, don't plot SH17 helmet boxes to avoid confusion
        plot_frame = frame.copy()
        
        for box in results.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # Apply class-specific thresholds
            req_conf = self.vest_conf if cls == VEST_CLASS_ID else self.min_conf
            if conf < req_conf:
                continue
                
            worker_id = int(box.id[0]) if box.id is not None else None
            
            x_c, y_c, w, h = float(box.xywhn[0][0]), float(box.xywhn[0][1]), float(box.xywhn[0][2]), float(box.xywhn[0][3])
            labels.append(f"{cls} {x_c} {y_c} {w} {h}")
            
            x1, y1, x2, y2 = float(box.xyxy[0][0]), float(box.xyxy[0][1]), float(box.xyxy[0][2]), float(box.xyxy[0][3])
            obj = {"cls": cls, "id": worker_id, "box": (x1, y1, x2, y2), "conf": conf}
            
            if cls == PERSON_CLASS_ID:
                persons.append(obj)
                # Draw person box manually
                cv2.rectangle(plot_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                id_txt = f"ID: {worker_id}" if worker_id else "Person"
                cv2.putText(plot_frame, id_txt, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                items.append(obj)
                # Draw items if not in two-stage helmet mode
                if not (self.ppe_detector_mode == "two_stage" and cls == HELMET_CLASS_ID):
                    color = (0, 255, 255) if cls == VEST_CLASS_ID else (255, 0, 255) if cls == GLOVES_CLASS_ID else (200, 200, 200)
                    cv2.rectangle(plot_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(plot_frame, f"{names[cls]}", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        geo_poly = Polygon(self.geofence_polygon) if len(self.geofence_polygon) >= 3 else None

        for person in persons:
            px1, py1, px2, py2 = person["box"]
            
            found_items = set()
            debug_state = {
                "worker_id": person["id"],
                "person_conf": person["conf"],
                "helmet": {"raw": False, "associated": False, "conf": 0.0},
                "vest": {"raw": False, "associated": False, "conf": 0.0},
                "gloves": {"raw": False, "associated": False, "conf": 0.0}
            }
            
            # Evaluate SH17 logic for Vests and Gloves
            for item in items:
                # We skip SH17 Helmet and Vest logic if in two_stage mode
                if self.ppe_detector_mode == "two_stage" and item["cls"] in [HELMET_CLASS_ID, VEST_CLASS_ID]:
                    print(f"DEBUG: Skipping SH17 logic for cls {item['cls']}", flush=True)
                    continue
                    
                if item["cls"] == HELMET_CLASS_ID:
                    debug_state["helmet"]["raw"] = True
                    debug_state["helmet"]["conf"] = max(debug_state["helmet"]["conf"], item["conf"])
                elif item["cls"] == VEST_CLASS_ID:
                    debug_state["vest"]["raw"] = True
                    debug_state["vest"]["conf"] = max(debug_state["vest"]["conf"], item["conf"])
                elif item["cls"] == GLOVES_CLASS_ID:
                    debug_state["gloves"]["raw"] = True
                    debug_state["gloves"]["conf"] = max(debug_state["gloves"]["conf"], item["conf"])

                ix1, iy1, ix2, iy2 = item["box"]
                inter_x1 = max(px1, ix1)
                inter_y1 = max(py1, iy1)
                inter_x2 = min(px2, ix2)
                inter_y2 = min(py2, iy2)
                
                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    item_area = (ix2 - ix1) * (iy2 - iy1)
                    if inter_area / item_area > 0.5:
                        if item["cls"] == HELMET_CLASS_ID: 
                            found_items.add("helmet")
                            debug_state["helmet"]["associated"] = True
                        elif item["cls"] == VEST_CLASS_ID: 
                            found_items.add("safety_vest")
                            debug_state["vest"]["associated"] = True
                        elif item["cls"] == GLOVES_CLASS_ID: 
                            found_items.add("gloves")
                            debug_state["gloves"]["associated"] = True

            # Two-Stage Helmet Detection
            if self.ppe_detector_mode == "two_stage" and self.helmet_model is not None:
                bw, bh = px2 - px1, py2 - py1
                ex1 = max(0, int(px1 - (bw * self.helmet_crop_expansion / 2)))
                ey1 = max(0, int(py1 - (bh * self.helmet_crop_expansion / 2)))
                ex2 = min(W, int(px2 + (bw * self.helmet_crop_expansion / 2)))
                ey2 = min(H, int(py2 + (bh * self.helmet_crop_expansion / 2)))
                
                if ex2 > ex1 and ey2 > ey1:
                    crop = frame[ey1:ey2, ex1:ex2]
                    
                    with self.model_lock:
                        device = '0' if sys.platform == 'win32' else 'cpu'
                        h_res = self.helmet_model.predict(crop, conf=self.helmet_conf, verbose=False, device=device)[0]
                        
                    if h_res.boxes is not None:
                        for h_box in h_res.boxes:
                            if int(h_box.cls[0]) == 0:  # Stage 2 helmet class is 0
                                found_items.add("helmet")
                                # DO NOT set debug_state["helmet"]["associated"] = True, that belongs to SH17
                                debug_state["helmet"]["stage2_conf"] = float(h_box.conf[0])
                                
                                # Draw box transformed to full frame
                                hx1, hy1, hx2, hy2 = float(h_box.xyxy[0][0]), float(h_box.xyxy[0][1]), float(h_box.xyxy[0][2]), float(h_box.xyxy[0][3])
                                fx1, fy1, fx2, fy2 = int(hx1 + ex1), int(hy1 + ey1), int(hx2 + ex1), int(hy2 + ey1)
                                cv2.rectangle(plot_frame, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)
                                cv2.putText(plot_frame, f"Helmet [HardHat] {debug_state['helmet']['conf']:.2f}", (fx1, max(fy1-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                                break # Associate the first valid helmet with this worker and move on

            # Two-Stage Vest Detection
            if self.ppe_detector_mode == "two_stage" and self.vest_model is not None:
                bw, bh = px2 - px1, py2 - py1
                ex1 = max(0, int(px1 - (bw * self.vest_crop_expansion / 2)))
                ey1 = max(0, int(py1 - (bh * self.vest_crop_expansion / 2)))
                ex2 = min(W, int(px2 + (bw * self.vest_crop_expansion / 2)))
                ey2 = min(H, int(py2 + (bh * self.vest_crop_expansion / 2)))
                
                if ex2 > ex1 and ey2 > ey1:
                    crop = frame[ey1:ey2, ex1:ex2]
                    
                    with self.model_lock:
                        device = '0' if sys.platform == 'win32' else 'cpu'
                        v_res = self.vest_model.predict(crop, conf=self.vest_conf, verbose=False, device=device)[0]
                        
                    if v_res.boxes is not None:
                        for v_box in v_res.boxes:
                            if int(v_box.cls[0]) == 0:  # Stage 2 vest class is 0
                                found_items.add("safety_vest")
                                # DO NOT set debug_state["vest"]["associated"] = True, that belongs to SH17
                                debug_state["vest"]["stage2_conf"] = float(v_box.conf[0])
                                
                                # Draw box transformed to full frame
                                vx1, vy1, vx2, vy2 = float(v_box.xyxy[0][0]), float(v_box.xyxy[0][1]), float(v_box.xyxy[0][2]), float(v_box.xyxy[0][3])
                                fx1, fy1, fx2, fy2 = int(vx1 + ex1), int(vy1 + ey1), int(vx2 + ex1), int(vy2 + ey1)
                                cv2.rectangle(plot_frame, (fx1, fy1), (fx2, fy2), (0, 255, 255), 2)
                                cv2.putText(plot_frame, f"Vest {debug_state['vest']['conf']:.2f}", (fx1, max(fy1-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                                break # Associate the first valid vest with this worker and move on


            missing = []
            if "helmet" not in found_items: missing.append("helmet")
            if "safety_vest" not in found_items: missing.append("safety vest")
            
            sh17_helmet_found = debug_state["helmet"]["associated"]
            sh17_vest_found = debug_state["vest"]["associated"]
            sh17_gloves_found = debug_state["gloves"]["associated"]
            final_helmet_found = "helmet" in found_items
            final_vest_found = "safety_vest" in found_items
            # Stage 2 helmet was found if it's in found_items but wasn't found by SH17
            stage2_helmet_found = final_helmet_found and not sh17_helmet_found
            stage2_vest_found = final_vest_found and not sh17_vest_found
            
            # Print verbose debugging information for the new integrated state
            print(f"[DEMO PPE STATE] Worker {person['id']}:", flush=True)
            print(f"  Person conf: {person['conf']:.2f}, Expansion: {self.vest_crop_expansion * 100}%", flush=True)
            print(f"  Helmet -> SH17: {sh17_helmet_found}, Stage-2: {stage2_helmet_found}, Final: {final_helmet_found}, Conf: {debug_state['helmet'].get('stage2_conf', debug_state['helmet']['conf']):.2f}", flush=True)
            print(f"  Vest   -> SH17: {sh17_vest_found}, Stage-2: {stage2_vest_found}, Final: {final_vest_found}, Conf: {debug_state['vest'].get('stage2_conf', debug_state['vest']['conf']):.2f}", flush=True)
            print(f"  Gloves -> SH17: {sh17_gloves_found}, Conf: {debug_state['gloves']['conf']:.2f}", flush=True)

            if missing:
                violation_found = True
                details = {"missing_items": missing}
                self.alert_manager.send_alert(self.client_id, "Missing PPE", person["conf"], worker_id=person["id"], details=details)
                
            if geo_poly:
                person_box = shapely_box(px1, py1, px2, py2)
                if person_box.intersects(geo_poly):
                    violation_found = True
                    details = "Worker entered restricted zone"
                    self.alert_manager.send_alert(self.client_id, "Zone Intrusion", person["conf"], worker_id=person["id"], details=details)

        if violation_found:
            os.makedirs("data/local_training/images", exist_ok=True)
            os.makedirs("data/local_training/labels", exist_ok=True)
            timestamp = int(time.time() * 1000)
            img_path = f"data/local_training/images/frame_{timestamp}.jpg"
            lbl_path = f"data/local_training/labels/frame_{timestamp}.txt"
            cv2.imwrite(img_path, frame)
            with open(lbl_path, "w") as f:
                f.write("\n".join(labels))
                
        if len(self.geofence_polygon) >= 3:
            pts = np.array(self.geofence_polygon, np.int32).reshape((-1, 1, 2))
            cv2.polylines(plot_frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

        return generate_saliency_map(plot_frame, self.model)
