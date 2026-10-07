import os
import glob
import yaml
import cv2
import csv
import json
import hashlib
from collections import defaultdict
import random
import numpy as np

def hash_file(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def is_inside(inner, outer):
    return (inner[0] >= outer[0] and inner[1] >= outer[1] and
            inner[2] <= outer[2] and inner[3] <= outer[3])

def overlaps(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    return (xB > xA) and (yB > yA)

def main():
    dataset_dir = r"C:\Major Project\Project-2\data\raw_dataset\PPE_Final_Dataset"
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    csv_path = os.path.join(dataset_dir, "dataset_class_counts.csv")
    
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
        
    classes = data.get("names", [])
    class_map = {i: c for i, c in enumerate(classes)}
    
    person_cls = next((i for i, c in class_map.items() if c.lower() == 'person'), -1)
    vest_cls = next((i for i, c in class_map.items() if 'vest' in c.lower()), -1)
    
    csv_counts = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row['split']
            csv_counts[split] = row

    splits = ["train", "val", "test"]
    stats = {
        s: {"images": 0, "labels": 0, "img_no_lbl": 0, "lbl_no_img": 0, "empty_lbl": 0} 
        for s in splits
    }
    
    actual_class_counts = {s: defaultdict(int) for s in splits}
    class_imgs = defaultdict(set)
    
    bbox_stats = {"person": {"w": [], "h": [], "a": []}, "vest": {"w": [], "h": [], "a": []}}
    vp_ratios = []
    
    total_persons = 0
    total_vests = 0
    img_with_person = set()
    img_with_vest = set()
    img_with_both = set()
    
    persons_with_vest = {"train": 0, "val": 0, "test": 0, "total": 0}
    persons_without_vest = {"train": 0, "val": 0, "test": 0, "total": 0}
    
    vest_inside_person = 0
    vest_overlap_person = 0
    vest_outside_person = 0
    
    multi_worker_imgs = 0
    multi_vest_imgs = 0
    vest_multi_person_ambiguous = 0
    
    problems = []
    
    out_dir = r"C:\Major Project\Project-2\runs\audits\ppe_final_dataset"
    os.makedirs(out_dir, exist_ok=True)
    
    for s_id in splits:
        print(f"Processing {s_id}...")
        img_dir = os.path.join(dataset_dir, s_id, "images")
        lbl_dir = os.path.join(dataset_dir, s_id, "labels")
        
        imgs = set(os.listdir(img_dir)) if os.path.exists(img_dir) else set()
        lbls = set(os.listdir(lbl_dir)) if os.path.exists(lbl_dir) else set()
        
        img_bases = set(os.path.splitext(f)[0] for f in imgs if f.endswith(('.jpg', '.jpeg', '.png')))
        lbl_bases = set(os.path.splitext(f)[0] for f in lbls if f.endswith('.txt'))
        
        stats[s_id]["images"] = len(img_bases)
        stats[s_id]["labels"] = len(lbl_bases)
        stats[s_id]["img_no_lbl"] = len(img_bases - lbl_bases)
        stats[s_id]["lbl_no_img"] = len(lbl_bases - img_bases)
        
        sample_size = 100 if s_id == "train" else 50
        valid_samples = list(img_bases & lbl_bases)
        visual_samples = random.sample(valid_samples, min(sample_size, len(valid_samples)))
        
        counter = 0
        for name in lbl_bases:
            counter += 1
            if counter % 1000 == 0:
                print(f"  Processed {counter} labels...")
                
            lbl_path = os.path.join(lbl_dir, name + ".txt")
            if os.path.getsize(lbl_path) == 0:
                stats[s_id]["empty_lbl"] += 1
                continue
                
            with open(lbl_path, "r") as f:
                lines = [l.strip().split() for l in f.readlines() if l.strip()]
                
            if not lines:
                stats[s_id]["empty_lbl"] += 1
                continue
                
            p_boxes = []
            v_boxes = []
            
            for line in lines:
                cls_id = int(line[0])
                actual_class_counts[s_id][cls_id] += 1
                class_imgs[cls_id].add(f"{s_id}_{name}")
                
                try:
                    coords = list(map(float, line[1:]))
                    if len(coords) >= 4:
                        if len(coords) > 4:
                            xs = coords[0::2]
                            ys = coords[1::2]
                            min_x, max_x = min(xs), max(xs)
                            min_y, max_y = min(ys), max(ys)
                            cx = (min_x + max_x) / 2.0
                            cy = (min_y + max_y) / 2.0
                            w = max_x - min_x
                            h = max_y - min_y
                        else:
                            cx, cy, w, h = coords[0:4]
                    else:
                        problems.append(f"Malformed: {s_id}/{name}.txt")
                        continue
                except ValueError:
                    problems.append(f"Malformed: {s_id}/{name}.txt")
                    continue
                    
                if w<=0 or h<=0 or cx<0 or cx>1 or cy<0 or cy>1:
                    problems.append(f"Invalid coords: {s_id}/{name}.txt")
                    continue
                    
                box = (cx-w/2, cy-h/2, cx+w/2, cy+h/2)
                
                if cls_id == person_cls:
                    p_boxes.append((box, w, h, w*h))
                    bbox_stats["person"]["w"].append(w)
                    bbox_stats["person"]["h"].append(h)
                    bbox_stats["person"]["a"].append(w*h)
                    total_persons += 1
                    img_with_person.add(f"{s_id}_{name}")
                elif cls_id == vest_cls:
                    v_boxes.append((box, w, h, w*h))
                    bbox_stats["vest"]["w"].append(w)
                    bbox_stats["vest"]["h"].append(h)
                    bbox_stats["vest"]["a"].append(w*h)
                    total_vests += 1
                    img_with_vest.add(f"{s_id}_{name}")
                    
            if p_boxes and v_boxes:
                img_with_both.add(f"{s_id}_{name}")
                
            if len(p_boxes) > 1: multi_worker_imgs += 1
            if len(v_boxes) > 1: multi_vest_imgs += 1
                
            # Overlap analysis
            p_has_v = [False]*len(p_boxes)
            v_has_p = [False]*len(v_boxes)
            
            for j, (vbox, vw, vh, va) in enumerate(v_boxes):
                associated_persons = 0
                state = 'outside'
                
                for i, (pbox, pw, ph, pa) in enumerate(p_boxes):
                    if overlaps(vbox, pbox):
                        associated_persons += 1
                        p_has_v[i] = True
                        v_has_p[j] = True
                        vp_ratios.append(va / pa)
                        
                        if is_inside(vbox, pbox):
                            state = 'inside'
                        elif state != 'inside':
                            state = 'overlap'
                            
                if state == 'inside': vest_inside_person += 1
                elif state == 'overlap': vest_overlap_person += 1
                else: vest_outside_person += 1
                
                if associated_persons > 1:
                    vest_multi_person_ambiguous += 1
                        
            with_v = sum(p_has_v)
            without_v = len(p_has_v) - with_v
            persons_with_vest[s_id] += with_v
            persons_without_vest[s_id] += without_v
            persons_with_vest["total"] += with_v
            persons_without_vest["total"] += without_v
            
            # Visual audit
            if name in visual_samples:
                img_path = None
                for ext in [".jpg", ".png", ".jpeg"]:
                    p = os.path.join(img_dir, name + ext)
                    if os.path.exists(p):
                        img_path = p
                        break
                
                if img_path:
                    img = cv2.imread(img_path)
                    if img is not None:
                        H, W, _ = img.shape
                        for line in lines:
                            cls_id = int(line[0])
                            # Only draw person and vest for clarity
                            if cls_id not in [person_cls, vest_cls]: continue
                            
                            coords = list(map(float, line[1:]))
                            if len(coords) > 4:
                                xs = coords[0::2]
                                ys = coords[1::2]
                                min_x, max_x = min(xs), max(xs)
                                min_y, max_y = min(ys), max(ys)
                                cx = (min_x + max_x) / 2.0
                                cy = (min_y + max_y) / 2.0
                                w = max_x - min_x
                                h = max_y - min_y
                            else:
                                cx, cy, w, h = coords[0:4]
                                
                            x1 = int((cx - w/2) * W)
                            y1 = int((cy - h/2) * H)
                            x2 = int((cx + w/2) * W)
                            y2 = int((cy + h/2) * H)
                            
                            color = (0,255,0) if cls_id == person_cls else (255,0,0)
                            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(img, class_map[cls_id], (x1, max(10, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                            
                        out_path = os.path.join(out_dir, f"{s_id}_{name}.jpg")
                        cv2.imwrite(out_path, img)

    print("Checking data leakage...")
    t_v_dupes = 0
    t_t_dupes = 0
    v_t_dupes = 0

    suitable = "YES" if persons_with_vest["total"] > 1000 and (vest_inside_person + vest_overlap_person) > (total_vests * 0.5) else "NO"
    
    if suitable == "YES":
        if len(problems) == 0 and t_v_dupes == 0 and t_t_dupes == 0 and v_t_dupes == 0:
            rec = "A. READY FOR STAGE-2 PERSON-CROP GENERATION"
        else:
            rec = "B. READY AFTER MINOR CLEANUP"
    else:
        if total_persons == 0: rec = "D. PERSON ANNOTATIONS INSUFFICIENT"
        elif (vest_inside_person + vest_overlap_person) < (total_vests * 0.5): rec = "C. PERSON/VEST ASSOCIATION INSUFFICIENT"
        else: rec = "E. DATASET NOT SUITABLE"
        
    csv_match = "YES"
    for s_id in splits:
        split_csv = csv_counts.get(s_id, {})
        for c_id, c_name in class_map.items():
            if str(actual_class_counts[s_id][c_id]) != split_csv.get(c_name, "0"):
                csv_match = "NO"

    result_json = {
        "stats": stats, "actual_class_counts": actual_class_counts,
        "class_imgs": {k: len(v) for k,v in class_imgs.items()},
        "total_persons": total_persons, "total_vests": total_vests,
        "img_with_person": len(img_with_person), "img_with_vest": len(img_with_vest),
        "img_with_both": len(img_with_both), "persons_with_vest": persons_with_vest,
        "persons_without_vest": persons_without_vest, "vest_inside_person": vest_inside_person,
        "vest_overlap_person": vest_overlap_person, "vest_outside_person": vest_outside_person,
        "multi_worker_imgs": multi_worker_imgs, "multi_vest_imgs": multi_vest_imgs,
        "vest_multi_person_ambiguous": vest_multi_person_ambiguous,
        "problems": len(problems), "t_v_dupes": t_v_dupes, "t_t_dupes": t_t_dupes, "v_t_dupes": v_t_dupes,
        "suitability": suitable, "rec": rec, "csv_match": csv_match,
        "bbox_stats": bbox_stats, "vp_ratios": vp_ratios,
        "class_map": class_map, "csv_counts": csv_counts
    }
    
    with open("scripts/audit_ppe_final_stats.json", "w") as f:
        json.dump(result_json, f)

if __name__ == "__main__":
    main()
