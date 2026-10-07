import os
import cv2
import json
import numpy as np
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(os.path.abspath('edge'))
from ultralytics import YOLO

# Mappings (to be verified)
EXPECTED_CLASSES = {"Person": 0, "Head": 1, "Gloves": 9, "Safety-vest": 12, "Helmet": 14}
RESULTS_DIR = Path("results/sh17_label_audit")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "train_samples").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "val_samples").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "target_class_audit").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "target_video_raw").mkdir(parents=True, exist_ok=True)

DS_PATH = Path(r"C:\Major Project\Results\datasets\SH17")
IMAGES_DIR = DS_PATH / "images"
LABELS_DIR = DS_PATH / "labels"

def get_iou(box1, box2):
    # box is [cx, cy, w, h] normalized
    x1_min, y1_min = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    x1_max, y1_max = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    x2_min, y2_min = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    x2_max, y2_max = box2[0] + box2[2]/2, box2[1] + box2[3]/2
    
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    if inter_xmax > inter_xmin and inter_ymax > inter_ymin:
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0
    return 0

def run_audit():
    report = ["# SH17 Label and Inference Audit\n"]
    
    # 1. VERIFY SH17 CLASS MAPPING
    report.append("## 1. Class Mappings")
    models = {
        "v8n": YOLO(r"edge\weights\sh17_yolov8n_best.pt"),
        "v8s": YOLO(r"edge\weights\sh17_yolov8s_best.pt")
    }
    
    for name, model in models.items():
        report.append(f"\n### {name} model.names:")
        for k, v in model.names.items():
            if v in EXPECTED_CLASSES:
                report.append(f"- **{v}**: {k} (Expected: {EXPECTED_CLASSES[v]})")
                
    # 2 & 3. RAW YOLO INFERENCE & VISUALIZATION (Indianworkers frame 100)
    report.append("\n## 2 & 3. Raw YOLO Inference (indianworkers.mp4, frame 100)")
    cap = cv2.VideoCapture(r"data\raw_dataset\source_files\source_files\indianworkers.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    ret, frame_100 = cap.read()
    cap.release()
    
    if ret:
        res = models["v8s"](frame_100, conf=0.45, verbose=False)[0]
        report.append("Raw Detections:")
        report.append("| Class | Confidence | Box |")
        report.append("|---|---|---|")
        
        vis_frame = frame_100.copy()
        
        for box in res.boxes:
            cls_id = int(box.cls[0])
            c_name = models["v8s"].names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            if c_name in EXPECTED_CLASSES:
                report.append(f"| {c_name} | {conf:.3f} | [{x1}, {y1}, {x2}, {y2}] |")
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(vis_frame, f"{c_name} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
        cv2.imwrite(str(RESULTS_DIR / "raw_frame_0100.jpg"), vis_frame)
        report.append("\n*Image saved to results/sh17_label_audit/raw_frame_0100.jpg*")
        
    # 4, 5, 8, 9. DATASET AUDIT
    report.append("\n## 4, 5, 8, 9. Dataset Audit")
    
    splits = {"train": [], "val": []}
    with open(DS_PATH / "train_files.txt", "r") as f:
        splits["train"] = [line.strip() for line in f.readlines() if line.strip()]
    with open(DS_PATH / "val_files.txt", "r") as f:
        splits["val"] = [line.strip() for line in f.readlines() if line.strip()]
        
    stats = {
        "Person": {"count": 0},
        "Head": {"count": 0, "w": [], "h": [], "cx": [], "cy": []},
        "Helmet": {"count": 0, "w": [], "h": [], "cx": [], "cy": []},
        "Safety-vest": {"count": 0, "w": [], "h": [], "cx": [], "cy": []},
        "Gloves": {"count": 0}
    }
    
    split_counts = {
        "train": {c: 0 for c in stats},
        "val": {c: 0 for c in stats}
    }
    
    iou_vest_helmet = []
    iou_vest_head = []
    
    format_errors = 0
    
    class_name_to_id = {v: k for k, v in EXPECTED_CLASSES.items()}
    class_id_to_name = EXPECTED_CLASSES
    # Wait, EXPECTED_CLASSES is Name -> ID. So we need ID -> Name
    id_to_name = {v: k for k, v in EXPECTED_CLASSES.items()}
    
    images_with_helmet_vest = []
    images_with_head_vest = []
    
    all_images = []
    
    for split_name, files in splits.items():
        for fname in files:
            base = os.path.splitext(fname)[0]
            label_path = LABELS_DIR / f"{base}.txt"
            img_path = IMAGES_DIR / fname
            if not label_path.exists(): continue
            
            all_images.append((split_name, fname))
            
            with open(label_path, "r") as f:
                lines = f.readlines()
                
            boxes_by_class = defaultdict(list)
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    format_errors += 1
                    continue
                c_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
                
                if cx < 0 or cx > 1 or cy < 0 or cy > 1 or w <= 0 or h <= 0:
                    format_errors += 1
                    continue
                    
                if c_id in id_to_name:
                    c_name = id_to_name[c_id]
                    boxes_by_class[c_name].append([cx, cy, w, h])
                    split_counts[split_name][c_name] += 1
                    stats[c_name]["count"] += 1
                    
                    if c_name in ["Head", "Helmet", "Safety-vest"]:
                        stats[c_name]["w"].append(w)
                        stats[c_name]["h"].append(h)
                        stats[c_name]["cx"].append(cx)
                        stats[c_name]["cy"].append(cy)
                        
            if "Safety-vest" in boxes_by_class and "Helmet" in boxes_by_class:
                images_with_helmet_vest.append(fname)
                for vest in boxes_by_class["Safety-vest"]:
                    for helmet in boxes_by_class["Helmet"]:
                        iou_vest_helmet.append(get_iou(vest, helmet))
                        
            if "Safety-vest" in boxes_by_class and "Head" in boxes_by_class:
                images_with_head_vest.append(fname)
                for vest in boxes_by_class["Safety-vest"]:
                    for head in boxes_by_class["Head"]:
                        iou_vest_head.append(get_iou(vest, head))
                        
    report.append(f"- Format Errors: {format_errors}")
    report.append("\n### Class Distribution")
    report.append("| Class | Train | Val | Total |")
    report.append("|---|---|---|---|")
    for c in stats:
        report.append(f"| {c} | {split_counts['train'][c]} | {split_counts['val'][c]} | {stats[c]['count']} |")
        
    report.append("\n### Bounding Box Statistics")
    report.append("| Class | Mean W | Mean H | Mean CX | Mean CY |")
    report.append("|---|---|---|---|---|")
    for c in ["Safety-vest", "Helmet", "Head"]:
        if stats[c]["count"] > 0:
            mw = np.mean(stats[c]["w"])
            mh = np.mean(stats[c]["h"])
            mcx = np.mean(stats[c]["cx"])
            mcy = np.mean(stats[c]["cy"])
            report.append(f"| {c} | {mw:.3f} | {mh:.3f} | {mcx:.3f} | {mcy:.3f} |")
            
    report.append("\n### Spatial Overlap (IoU)")
    def iou_stats(ious, name):
        if not ious: return f"{name}: No pairs found."
        miou = np.mean(ious)
        mediou = np.median(ious)
        maxiou = np.max(ious)
        p30 = sum(1 for i in ious if i > 0.3) / len(ious) * 100
        p50 = sum(1 for i in ious if i > 0.5) / len(ious) * 100
        return f"- **{name}**: Mean: {miou:.3f}, Median: {mediou:.3f}, Max: {maxiou:.3f} | >0.3: {p30:.1f}% | >0.5: {p50:.1f}%"
        
    report.append(iou_stats(iou_vest_helmet, "Safety-vest & Helmet"))
    report.append(iou_stats(iou_vest_head, "Safety-vest & Head"))
    
    # 6. RANDOM VISUAL LABEL AUDIT
    train_samples = random.sample([f for s, f in all_images if s == 'train'], min(50, len([f for s, f in all_images if s == 'train'])))
    val_samples = random.sample([f for s, f in all_images if s == 'val'], min(50, len([f for s, f in all_images if s == 'val'])))
    
    def draw_gt(img_name, out_dir):
        img_path = IMAGES_DIR / img_name
        label_path = LABELS_DIR / f"{os.path.splitext(img_name)[0]}.txt"
        img = cv2.imread(str(img_path))
        if img is None: return
        H, W = img.shape[:2]
        
        with open(label_path, "r") as f:
            lines = f.readlines()
            
        colors = {"Helmet": (255, 0, 0), "Head": (0, 255, 0), "Safety-vest": (0, 0, 255)}
        for line in lines:
            parts = line.strip().split()
            c_id = int(parts[0])
            if c_id in id_to_name:
                c_name = id_to_name[c_id]
                if c_name in colors:
                    cx, cy, w, h = map(float, parts[1:])
                    x1 = int((cx - w/2) * W)
                    y1 = int((cy - h/2) * H)
                    x2 = int((cx + w/2) * W)
                    y2 = int((cy + h/2) * H)
                    cv2.rectangle(img, (x1, y1), (x2, y2), colors[c_name], 2)
                    cv2.putText(img, c_name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[c_name], 2)
        cv2.imwrite(str(out_dir / img_name), img)
        
    for fname in train_samples: draw_gt(fname, RESULTS_DIR / "train_samples")
    for fname in val_samples: draw_gt(fname, RESULTS_DIR / "val_samples")
    
    # Target Class Audit
    target_vest_helmet = random.sample(images_with_helmet_vest, min(50, len(images_with_helmet_vest)))
    target_vest_head = random.sample(images_with_head_vest, min(50, len(images_with_head_vest)))
    
    for fname in target_vest_helmet: draw_gt(fname, RESULTS_DIR / "target_class_audit")
    for fname in target_vest_head: draw_gt(fname, RESULTS_DIR / "target_class_audit")
    
    # 11. TARGET VIDEO RAW ANALYSIS
    videos_to_test = [
        (r"data\raw_dataset\source_files\source_files\JapanPPE.mp4", 300),
        (r"data\raw_dataset\source_files\source_files\hardhat.mp4", 100),
        (r"data\raw_dataset\source_files\source_files\indianworkers.mp4", 100)
    ]
    
    for v_path, f_idx in videos_to_test:
        v_name = os.path.basename(v_path)
        cap = cv2.VideoCapture(v_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            res = models["v8s"](frame, conf=0.45, verbose=False)[0]
            vis_frame = frame.copy()
            for box in res.boxes:
                cls_id = int(box.cls[0])
                c_name = models["v8s"].names[cls_id]
                if c_name in ["Safety-vest", "Helmet", "Head"]:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(vis_frame, c_name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.imwrite(str(RESULTS_DIR / "target_video_raw" / f"{v_name}_raw_frame.jpg"), vis_frame)
            
    # Write report
    with open(RESULTS_DIR / "sh17_label_audit.md", "w") as f:
        f.write("\n".join(report))
        
    print("Audit Complete!")

if __name__ == "__main__":
    run_audit()
