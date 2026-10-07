import os
import glob
import json
import cv2
import random
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

random.seed(42)

def hash_file(filepath):
    # Lightweight hash (md5 of the first 64k) to speed up duplicate check
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        hasher.update(buf)
        # Also incorporate file size
        hasher.update(str(os.path.getsize(filepath)).encode())
    return hasher.hexdigest()

def run_audit():
    DS_DIR = Path(r"C:\Major Project\Project-2\data\raw_dataset\HardHat-Vest Dataset")
    OUT_DIR = Path(r"C:\Major Project\Project-2\reports\hardhat_vest_audit")
    
    for d in ["helmet_samples"]:
        (OUT_DIR / d).mkdir(parents=True, exist_ok=True)
        
    splits = ["train", "val", "test"]
    
    classes_file = DS_DIR / "labels" / "classes.txt"
    class_names = {}
    if classes_file.exists():
        with open(classes_file, 'r') as f:
            for i, line in enumerate(f):
                if line.strip():
                    class_names[i] = line.strip()
    else:
        class_names = {0: "helmet", 1: "vest", 2: "head", 3: "person"}
        
    stats = {
        "images": {"train": 0, "val": 0, "test": 0, "total": 0},
        "labels": {"train": 0, "val": 0, "test": 0, "total": 0},
        "missing_images": 0,
        "missing_labels": 0,
        "empty_labels": 0,
        "malformed_lines": 0,
        "out_of_bounds": 0,
        "zero_area": 0,
        "invalid_class": 0,
        "formats": set(),
        "resolutions": set()
    }
    
    class_dist = defaultdict(lambda: {"train": 0, "val": 0, "test": 0, "total": 0, "images": set()})
    
    helmet_images = []
    file_hashes = {}
    duplicates = []
    
    for split in splits:
        img_dir = DS_DIR / "images" / split
        lbl_dir = DS_DIR / "labels" / split
        
        if not img_dir.exists() or not lbl_dir.exists():
            continue
            
        imgs = list(img_dir.glob("*.*"))
        lbls = list(lbl_dir.glob("*.txt"))
        
        img_stems = {f.stem: f for f in imgs}
        lbl_stems = {f.stem: f for f in lbls}
        
        stats["images"][split] = len(imgs)
        stats["labels"][split] = len(lbls)
        stats["images"]["total"] += len(imgs)
        stats["labels"]["total"] += len(lbls)
        
        for f in imgs:
            stats["formats"].add(f.suffix.lower())
            
        for stem, img_path in img_stems.items():
            if stem not in lbl_stems:
                stats["missing_labels"] += 1
            else:
                lbl_path = lbl_stems[stem]
                
                # Hash check for duplicates
                h = hash_file(img_path)
                if h in file_hashes:
                    duplicates.append((str(img_path), str(file_hashes[h])))
                else:
                    file_hashes[h] = img_path
                
                if os.path.getsize(lbl_path) == 0:
                    stats["empty_labels"] += 1
                    continue
                    
                has_helmet = False
                valid_boxes = []
                
                with open(lbl_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            stats["malformed_lines"] += 1
                            continue
                            
                        try:
                            c = int(parts[0])
                            cx, cy, w, h = map(float, parts[1:])
                            
                            if c not in class_names:
                                stats["invalid_class"] += 1
                                continue
                                
                            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                                stats["out_of_bounds"] += 1
                                continue
                                
                            if w <= 0 or h <= 0:
                                stats["zero_area"] += 1
                                continue
                                
                            class_dist[c][split] += 1
                            class_dist[c]["total"] += 1
                            class_dist[c]["images"].add(str(img_path))
                            
                            if c == 0: # helmet
                                has_helmet = True
                                
                            valid_boxes.append((c, cx, cy, w, h))
                                
                        except ValueError:
                            stats["malformed_lines"] += 1
                            
                if has_helmet:
                    helmet_images.append((img_path, valid_boxes))
                    
        for stem in lbl_stems:
            if stem not in img_stems:
                stats["missing_images"] += 1

    stats["formats"] = list(stats["formats"])
    
    # Save helmet samples
    print(f"Generating helmet samples from {len(helmet_images)} candidates...")
    sample_imgs = random.sample(helmet_images, min(100, len(helmet_images)))
    colors = {0: (255, 0, 0), 1: (0, 0, 255), 2: (0, 255, 0), 3: (0, 255, 255)}
    
    for i, (img_path, boxes) in enumerate(sample_imgs):
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        # sample resolution
        if len(stats["resolutions"]) < 20:
            stats["resolutions"].add(f"{img.shape[1]}x{img.shape[0]}")
            
        H, W = img.shape[:2]
        for c, cx, cy, w, h in boxes:
            name = class_names[c]
            col = colors.get(c, (255,255,255))
            x1 = int((cx - w/2) * W)
            y1 = int((cy - h/2) * H)
            x2 = int((cx + w/2) * W)
            y2 = int((cy + h/2) * H)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            cv2.putText(img, name, (x1, max(y1-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            
        out_name = f"sample_{i:03d}_{img_path.name}"
        cv2.imwrite(str(OUT_DIR / "helmet_samples" / out_name), img)

    stats["resolutions"] = list(stats["resolutions"])
    
    # Write summary
    with open(OUT_DIR / "dataset_summary.json", 'w') as f:
        json.dump(stats, f, indent=4)
        
    with open(OUT_DIR / "duplicate_report.json", 'w') as f:
        json.dump(duplicates, f, indent=4)
        
    # Write class dist CSV
    with open(OUT_DIR / "class_distribution.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Class Name", "Class ID", "Train Instances", "Val Instances", "Test Instances", "Total Instances", "Images Containing Class"])
        for c, d in class_dist.items():
            writer.writerow([class_names[c], c, d["train"], d["val"], d["test"], d["total"], len(d["images"])])

    # Annotation statistics
    with open(OUT_DIR / "annotation_statistics.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for k, v in stats.items():
            if not isinstance(v, (dict, list, set)):
                writer.writerow([k, v])
                
    print("Audit data generation complete.")

if __name__ == "__main__":
    run_audit()
