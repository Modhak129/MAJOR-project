import os
from ultralytics import YOLO
import json

def get_class_metrics(res):
    d = {}
    box = res.box
    if hasattr(box, 'ap_class_index') and hasattr(box, 'class_result'):
        for i, cls_idx in enumerate(box.ap_class_index):
            # precision, recall, map50, map50-95
            p, r, m50, m95 = box.class_result(i)
            d[cls_idx] = (p, r, m50, m95)
    return d

def main():
    print("Evaluating SH17 Model on SH17 Validation Set...")
    sh17 = YOLO("edge/weights/sh17_yolov8n_best.pt")
    
    # We will use the proper data.yaml from css-data if possible, else fallback
    css_data_yaml = r"C:\Major Project\Project-2\data\raw_dataset\css-data\valid\images"
    if os.path.exists(css_data_yaml):
        # We need to make sure paths are absolute. We'll write a quick local yaml.
        sh17_yaml = "sh17_absolute.yaml"
        with open(sh17_yaml, "w") as f:
            f.write(f"train: {os.path.abspath('data/raw_dataset/css-data/train/images')}\n")
            f.write(f"val: {os.path.abspath('data/raw_dataset/css-data/valid/images')}\n")
            f.write("nc: 17\n")
            f.write("names: {0: 'Person', 1: 'Head', 2: 'Face', 3: 'Glasses', 4: 'Face-mask-medical', 5: 'Face-guard', 6: 'Ear', 7: 'Earmuffs', 8: 'Hands', 9: 'Gloves', 10: 'Foot', 11: 'Shoes', 12: 'Safety-vest', 13: 'Tools', 14: 'Helmet', 15: 'Medical-suit', 16: 'Safety-suit'}\n")
        
        sh17_res = sh17.val(data=sh17_yaml, split="val", imgsz=640, device=0)
    else:
        print("Fallback to sh17_local.yaml")
        sh17_res = sh17.val(data="sh17_local.yaml", split="val", imgsz=640, device=0)
    
    print("\nEvaluating Roboflow Model on Roboflow Validation Set...")
    rf = YOLO("runs/detect/edge/weights/training/roboflow/roboflow_yolov8n_640/weights/best.pt")
    rf_res = rf.val(data="roboflow_local.yaml", split="val", imgsz=640, device=0)
    
    sh17_metrics = sh17_res.results_dict
    rf_metrics = rf_res.results_dict
    
    sh17_map = {0: "Person", 14: "Helmet", 12: "Safety Vest", 9: "Gloves"}
    rf_map = {11: "Person", 3: "Helmet", 13: "Safety Vest", 1: "Gloves"}
    
    sh17_cls = get_class_metrics(sh17_res)
    rf_cls = get_class_metrics(rf_res)
    
    table = f"""# Validation Comparison

Metric | SH17 YOLOv8n (css-data valid) | Roboflow YOLOv8n (Roboflow valid)
---|---|---
Overall Precision | {sh17_metrics['metrics/precision(B)']:.3f} | {rf_metrics['metrics/precision(B)']:.3f}
Overall Recall | {sh17_metrics['metrics/recall(B)']:.3f} | {rf_metrics['metrics/recall(B)']:.3f}
Overall mAP50 | {sh17_metrics['metrics/mAP50(B)']:.3f} | {rf_metrics['metrics/mAP50(B)']:.3f}
Overall mAP50-95 | {sh17_metrics['metrics/mAP50-95(B)']:.3f} | {rf_metrics['metrics/mAP50-95(B)']:.3f}

### Shared Semantic Classes (mAP50)

| Class | SH17 mAP50 | Roboflow mAP50 | SH17 Recall | Roboflow Recall |
|---|---|---|---|---|
"""
    for sh17_id, name in sh17_map.items():
        rf_id = next((k for k, v in rf_map.items() if v == name), None)
        
        sh17_stats = sh17_cls.get(sh17_id, [0,0,0,0])
        rf_stats = rf_cls.get(rf_id, [0,0,0,0]) if rf_id is not None else [0,0,0,0]
        
        table += f"| {name} | {sh17_stats[2]:.3f} | {rf_stats[2]:.3f} | {sh17_stats[1]:.3f} | {rf_stats[1]:.3f} |\n"
        
    os.makedirs("results/roboflow_vs_sh17", exist_ok=True)
    with open("results/roboflow_vs_sh17/validation_comparison.md", "w") as f:
        f.write(table)
        
    print("\n" + table)

if __name__ == "__main__":
    main()
