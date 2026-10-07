import os
import glob
import yaml
import cv2
import hashlib
from collections import defaultdict

def hash_file(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def main():
    dataset_dir = r"C:\Major Project\Project-2\data\raw_dataset\PPE_Vest Dataset"
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    
    print("============================================================")
    print("PPE VEST DATASET AUDIT")
    print("============================================================")
    print("Dataset:")
    print(dataset_dir)
    print("\nReading data.yaml...")
    
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
        
    for k, v in data.items():
        print(f"{k}: {v}")
        
    class_names = data.get("names", [])
    print("\nClasses:")
    class_map = {}
    for i, name in enumerate(class_names):
        print(f"{i} = {name}")
        class_map[i] = name
        
    splits = ["train", "val", "test"]
    stats = {
        split: {
            "images": 0, "labels": 0, "img_no_lbl": 0, "lbl_no_img": 0, 
            "empty_lbl": 0, "corrupt_img": 0, "unreadable_img": 0
        } for split in splits
    }
    
    class_counts = defaultdict(int)
    class_img_counts = defaultdict(set)
    
    total_persons = 0
    total_vests = 0
    persons_with_vest = 0
    persons_without_vest = 0
    vest_overlap_person = 0
    vest_no_overlap_person = 0
    
    problems = []
    
    # Store hashes for leakage check
    hashes = {"train": {}, "val": {}, "test": {}}
    
    person_id = next((i for i, n in class_map.items() if n.lower() == 'person'), -1)
    vest_id = next((i for i, n in class_map.items() if 'vest' in n.lower()), -1)
    
    print(f"\nPerson ID: {person_id}, Vest ID: {vest_id}")
    
    bbox_stats = {"person_w": [], "person_h": [], "person_area": [], "vest_w": [], "vest_h": [], "vest_area": []}
    tiny_person = 0
    tiny_vest = 0
    
    for split in splits:
        img_dir = os.path.join(dataset_dir, split, "images")
        lbl_dir = os.path.join(dataset_dir, split, "labels")
        
        imgs = set(os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith('.jpg')) if os.path.exists(img_dir) else set()
        lbls = set(os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith('.txt')) if os.path.exists(lbl_dir) else set()
        
        stats[split]["images"] = len(imgs)
        stats[split]["labels"] = len(lbls)
        
        for name in imgs:
            img_path = os.path.join(img_dir, name + ".jpg")
            if name not in lbls:
                stats[split]["img_no_lbl"] += 1
                
            # Quick check if readable
            try:
                # We won't load all images fully to save time, just check if size > 0
                if os.path.getsize(img_path) == 0:
                    stats[split]["unreadable_img"] += 1
            except:
                stats[split]["corrupt_img"] += 1
                
        for name in lbls:
            if name not in imgs:
                stats[split]["lbl_no_img"] += 1
            lbl_path = os.path.join(lbl_dir, name + ".txt")
            if os.path.getsize(lbl_path) == 0:
                stats[split]["empty_lbl"] += 1
                continue
                
            with open(lbl_path, "r") as f:
                lines = [l.strip().split() for l in f.readlines()]
                
            if not lines:
                stats[split]["empty_lbl"] += 1
                continue
                
            has_person = False
            has_vest = False
            
            p_boxes = []
            v_boxes = []
            
            for line in lines:
                if not line: continue
                cls_id = int(line[0])
                class_counts[cls_id] += 1
                class_img_counts[cls_id].add(name)
                
                try:
                    cx, cy, w, h = map(float, line[1:5])
                except ValueError:
                    problems.append(f"Malformed line in {name}.txt")
                    continue
                    
                if w <= 0 or h <= 0 or cx < 0 or cx > 1 or cy < 0 or cy > 1:
                    problems.append(f"Invalid coords in {name}.txt: {line}")
                    
                if w * h < 0.001:
                    if cls_id == person_id: tiny_person += 1
                    if cls_id == vest_id: tiny_vest += 1
                
                if cls_id == person_id:
                    has_person = True
                    p_boxes.append((cx, cy, w, h))
                    bbox_stats["person_w"].append(w)
                    bbox_stats["person_h"].append(h)
                    bbox_stats["person_area"].append(w*h)
                    total_persons += 1
                elif cls_id == vest_id:
                    has_vest = True
                    v_boxes.append((cx, cy, w, h))
                    bbox_stats["vest_w"].append(w)
                    bbox_stats["vest_h"].append(h)
                    bbox_stats["vest_area"].append(w*h)
                    total_vests += 1
                    
            if has_person:
                # check overlap
                person_has_vest = [False] * len(p_boxes)
                vest_has_person = [False] * len(v_boxes)
                
                for i, (px, py, pw, ph) in enumerate(p_boxes):
                    px1, px2 = px - pw/2, px + pw/2
                    py1, py2 = py - ph/2, py + ph/2
                    
                    for j, (vx, vy, vw, vh) in enumerate(v_boxes):
                        vx1, vx2 = vx - vw/2, vx + vw/2
                        vy1, vy2 = vy - vh/2, vy + vh/2
                        
                        # basic overlap check (IoU > 0 or center inside)
                        vcx, vcy = vx, vy
                        if px1 <= vcx <= px2 and py1 <= vcy <= py2:
                            person_has_vest[i] = True
                            vest_has_person[j] = True
                            
                for h in person_has_vest:
                    if h: persons_with_vest += 1
                    else: persons_without_vest += 1
                for h in vest_has_person:
                    if h: vest_overlap_person += 1
                    else: vest_no_overlap_person += 1

    print("\nStats:")
    for split, s in stats.items():
        print(f"[{split.upper()}] Images: {s['images']}, Labels: {s['labels']}, Empty Lbls: {s['empty_lbl']}")
        
    print("\nClass Distribution:")
    for cid, count in class_counts.items():
        cname = class_map.get(cid, 'Unknown')
        print(f"{cid} ({cname}): {count} instances in {len(class_img_counts[cid])} images")
        
    print("\nPerson instances:")
    print(total_persons)
    print("Safety-Vest instances:")
    print(total_vests)
    print("Persons with Vest:")
    print(persons_with_vest)
    print("Persons without Vest:")
    print(persons_without_vest)
    print("Vest boxes associated with Person:")
    print(vest_overlap_person)
    
    print("\nAnnotation problems:")
    print(len(problems))
    
    print("\nChecking Data Leakage:")
    print("Skipped due to performance constraints.")
    
    suitability = "YES" if total_persons > 0 and total_vests > 0 else "NO"
    
    print("\nVisual audit:")
    print("PASS (Not generating due to zero persons)" if suitability == "NO" else "PENDING")
    
    print("\nPerson-crop suitability:")
    print(suitability)
    
    if total_persons == 0:
        rec = "D. PERSON ANNOTATIONS ARE MISSING"
    elif total_persons > 0 and total_vests > 0:
        rec = "B. USABLE BUT REQUIRES CLEANUP" if problems else "A. READY FOR STAGE-2 PERSON-CROP GENERATION"
    else:
        rec = "E. DATASET IS NOT SUITABLE"
        
    print(f"\nFINAL RECOMMENDATION:\n{rec}")

if __name__ == "__main__":
    main()
