import os
import yaml
from pathlib import Path
from collections import defaultdict

def audit_dataset():
    dataset_dir = Path(r"C:\Major Project\Project-2\data\raw_dataset\Dataset Roboflow")
    
    with open(dataset_dir / "data.yaml", "r") as f:
        data = yaml.safe_load(f)
        
    names = data['names']
    nc = data['nc']
    
    print(f"Dataset has {nc} classes:")
    for i, name in enumerate(names):
        print(f"  {i}: {name}")
        
    stats = {
        "train": {"images": 0, "labels": 0, "empty_labels": 0, "annotations": 0, "class_counts": defaultdict(int), "malformed": 0, "oob": 0},
        "valid": {"images": 0, "labels": 0, "empty_labels": 0, "annotations": 0, "class_counts": defaultdict(int), "malformed": 0, "oob": 0},
        "test": {"images": 0, "labels": 0, "empty_labels": 0, "annotations": 0, "class_counts": defaultdict(int), "malformed": 0, "oob": 0}
    }
    
    for split in ["train", "valid", "test"]:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue
            
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"
        
        images = list(img_dir.glob("*.*"))
        stats[split]["images"] = len(images)
        
        for img_path in images:
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                stats[split]["empty_labels"] += 1
                continue
                
            stats[split]["labels"] += 1
            
            with open(lbl_path, "r") as f:
                lines = f.readlines()
                
            if not lines:
                stats[split]["empty_labels"] += 1
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    stats[split]["malformed"] += 1
                    continue
                    
                cls_id = int(parts[0])
                stats[split]["annotations"] += 1
                stats[split]["class_counts"][cls_id] += 1
                
                # Check bounds
                try:
                    cx, cy, w, h = map(float, parts[1:])
                    if cx < 0 or cx > 1 or cy < 0 or cy > 1 or w <= 0 or h <= 0:
                        stats[split]["oob"] += 1
                except:
                    stats[split]["malformed"] += 1
                    
    print("\n================ DATASET AUDIT ================")
    for split, data in stats.items():
        print(f"\nSplit: {split}")
        print(f"  Images: {data['images']}")
        print(f"  Labels: {data['labels']}")
        print(f"  Empty labels: {data['empty_labels']}")
        print(f"  Annotations: {data['annotations']}")
        print(f"  Malformed lines: {data['malformed']}")
        print(f"  Out of bounds/zero-size: {data['oob']}")
        print("  Class breakdown:")
        for cls_id in sorted(data["class_counts"].keys()):
            name = names[cls_id] if cls_id < nc else f"Unknown ({cls_id})"
            print(f"    {cls_id} ({name}): {data['class_counts'][cls_id]}")

if __name__ == "__main__":
    audit_dataset()
