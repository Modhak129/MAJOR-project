import os
import cv2
import csv
import json
import yaml
import shutil
import random
from collections import defaultdict

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def box_intersection(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    if xB <= xA or yB <= yA:
        return 0.0
    return (xB - xA) * (yB - yA)

def is_center_inside(box, container):
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    return (container[0] <= cx <= container[2]) and (container[1] <= cy <= container[3])

def get_area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

def main():
    dataset_dir = r"C:\Major Project\Project-2\data\raw_dataset\PPE_Final_Dataset"
    output_dir_base = r"C:\Major Project\Project-2\data\processed"
    audit_out_dir = r"C:\Major Project\Project-2\runs\audits\stage2_vest_crop_generation"
    ensure_dir(audit_out_dir)
    
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    
    classes = data.get("names", [])
    class_map = {i: c for i, c in enumerate(classes)}
    
    person_cls = next((i for i, c in class_map.items() if c.lower() == 'person'), -1)
    vest_cls = next((i for i, c in class_map.items() if 'vest' in c.lower()), -1)
    
    splits = ["train", "val", "test"]
    expansions = [0, 10, 15, 20, 25]
    
    # Prepare output directories
    for exp in expansions:
        exp_dir = os.path.join(output_dir_base, f"stage2_vest_crops_{exp}")
        if os.path.exists(exp_dir):
            shutil.rmtree(exp_dir)
        for split in splits:
            ensure_dir(os.path.join(exp_dir, split, "images"))
            ensure_dir(os.path.join(exp_dir, split, "labels"))
        
        yaml_content = {"train": "train/images", "val": "val/images", "test": "test/images", "nc": 1, "names": ["safety_vest"]}
        with open(os.path.join(exp_dir, "data.yaml"), "w") as f:
            yaml.dump(yaml_content, f, sort_keys=False)

    unassociated_csv = open(os.path.join(audit_out_dir, "unassociated_vests.csv"), "w", newline='')
    unassociated_writer = csv.writer(unassociated_csv)
    unassociated_writer.writerow(["image", "split", "vest_id", "vest_bbox", "reason"])
    
    ambiguous_csv = open(os.path.join(audit_out_dir, "ambiguous_vest_person_associations.csv"), "w", newline='')
    ambiguous_writer = csv.writer(ambiguous_csv)
    ambiguous_writer.writerow(["image", "split", "vest_id", "candidate_person_ids", "iou_values", "vest_inside_person_pct", "selected_person", "decision", "reason"])

    # Statistics tracking
    stats = {exp: {"pos": 0, "neg": 0, "ambiguous": 0, "discarded": 0, "tiny": 0, "normal": 0, "total": 0, 
                   "splits": {s: {"pos": 0, "neg": 0, "ambiguous": 0, "discarded": 0, "tiny": 0, "normal": 0, "total": 0} for s in splits}} 
             for exp in expansions}
    
    global_stats = {
        "total_persons": 0, "total_vests": 0,
        "positive_associations": 0, "negative_persons": 0,
        "ambiguous_associations": 0, "unassociated_vests": 0,
        "images_with_person": 0, "images_with_vest": 0, "images_with_both": 0,
        "ambiguous_persons": 0
    }
    
    validation_errors = []
    
    # Randomly select a few images for visualization per expansion
    vis_samples = {s: [] for s in splits}
    for split in splits:
        img_dir = os.path.join(dataset_dir, split, "images")
        if os.path.exists(img_dir):
            all_imgs = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            vis_samples[split] = set(random.sample(all_imgs, min(50, len(all_imgs))))
            
    for split in splits:
        print(f"Processing split: {split}")
        img_dir = os.path.join(dataset_dir, split, "images")
        lbl_dir = os.path.join(dataset_dir, split, "labels")
        if not os.path.exists(img_dir): continue
        
        img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        for img_idx, img_name in enumerate(img_files):
            if img_idx % 1000 == 0:
                print(f"  {img_idx}/{len(img_files)} images processed...")
                
            base_name = os.path.splitext(img_name)[0]
            lbl_path = os.path.join(lbl_dir, base_name + ".txt")
            img_path = os.path.join(img_dir, img_name)
            
            persons = []
            vests = []
            
            if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
                img = cv2.imread(img_path)
                if img is None: continue
                H, W, _ = img.shape
                
                with open(lbl_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts: continue
                        cls_id = int(parts[0])
                        if cls_id not in [person_cls, vest_cls]: continue
                        
                        cx, cy, w, h = map(float, parts[1:5])
                        x1 = (cx - w/2) * W
                        y1 = (cy - h/2) * H
                        x2 = (cx + w/2) * W
                        y2 = (cy + h/2) * H
                        
                        if cls_id == person_cls:
                            persons.append({'id': len(persons), 'box': (x1, y1, x2, y2), 'w': w*W, 'h': h*H, 'status': None, 'vests': []})
                            global_stats["total_persons"] += 1
                        elif cls_id == vest_cls:
                            vests.append({'id': len(vests), 'box': (x1, y1, x2, y2), 'area': (x2-x1)*(y2-y1)})
                            global_stats["total_vests"] += 1
                            
                has_p = len(persons) > 0
                has_v = len(vests) > 0
                if has_p: global_stats["images_with_person"] += 1
                if has_v: global_stats["images_with_vest"] += 1
                if has_p and has_v: global_stats["images_with_both"] += 1
                
                # ASSOCIATION
                ambiguous_p_ids = set()
                for v in vests:
                    candidates = []
                    for p in persons:
                        inter = box_intersection(v['box'], p['box'])
                        if inter > 0:
                            v_in_p = inter / v['area']
                            c_inside = is_center_inside(v['box'], p['box'])
                            candidates.append({
                                'p_id': p['id'],
                                'inter': inter,
                                'v_in_p': v_in_p,
                                'c_inside': c_inside,
                                'score': v_in_p + (0.5 if c_inside else 0)
                            })
                            
                    if not candidates:
                        global_stats["unassociated_vests"] += 1
                        unassociated_writer.writerow([img_name, split, v['id'], v['box'], "No overlap with any person"])
                    elif len(candidates) == 1:
                        c = candidates[0]
                        if c['v_in_p'] > 0.2 or c['c_inside']:
                            persons[c['p_id']]['vests'].append(v)
                        else:
                            global_stats["unassociated_vests"] += 1
                            unassociated_writer.writerow([img_name, split, v['id'], v['box'], f"Overlap too small: {c['v_in_p']:.2f}"])
                    else:
                        candidates.sort(key=lambda x: x['score'], reverse=True)
                        c1 = candidates[0]
                        c2 = candidates[1]
                        
                        if c1['score'] - c2['score'] > 0.3:
                            persons[c1['p_id']]['vests'].append(v)
                        else:
                            global_stats["ambiguous_associations"] += 1
                            ambiguous_p_ids.add(c1['p_id'])
                            ambiguous_p_ids.add(c2['p_id'])
                            c_ids = [c['p_id'] for c in candidates]
                            ious = [c['inter'] for c in candidates]
                            v_in_ps = [c['v_in_p'] for c in candidates]
                            ambiguous_writer.writerow([img_name, split, v['id'], c_ids, ious, v_in_ps, "NONE", "AMBIGUOUS", f"Score diff too small: {c1['score']:.2f} vs {c2['score']:.2f}"])
                            
                for p in persons:
                    if p['id'] in ambiguous_p_ids:
                        p['status'] = 'AMBIGUOUS'
                        global_stats["ambiguous_persons"] += 1
                    else:
                        is_positive = len(p['vests']) > 0
                        if is_positive:
                            global_stats["positive_associations"] += 1
                            p['status'] = 'POSITIVE'
                        else:
                            global_stats["negative_persons"] += 1
                            p['status'] = 'NEGATIVE'
                        
                # CROP GENERATION
                for p in persons:
                    px1, py1, px2, py2 = p['box']
                    pw = px2 - px1
                    ph = py2 - py1
                    
                    for exp in expansions:
                        if p['status'] == 'AMBIGUOUS':
                            stats[exp]["ambiguous"] += 1
                            stats[exp]["splits"][split]["ambiguous"] += 1
                            continue
                            
                        exp_factor = exp / 100.0
                        exp_w = pw * exp_factor
                        exp_h = ph * exp_factor
                        
                        nx1 = max(0, int(px1 - exp_w/2))
                        ny1 = max(0, int(py1 - exp_h/2))
                        nx2 = min(W, int(px2 + exp_w/2))
                        ny2 = min(H, int(py2 + exp_h/2))
                        
                        crop_w = nx2 - nx1
                        crop_h = ny2 - ny1
                        
                        if crop_w < 16 or crop_h < 16:
                            stats[exp]["discarded"] += 1
                            stats[exp]["splits"][split]["discarded"] += 1
                            continue
                            
                        size_cat = "tiny" if crop_w < 32 or crop_h < 32 else "normal"
                        stats[exp][size_cat] += 1
                        stats[exp]["splits"][split][size_cat] += 1
                        
                        if p['status'] == 'POSITIVE':
                            stats[exp]["pos"] += 1
                            stats[exp]["splits"][split]["pos"] += 1
                        else:
                            stats[exp]["neg"] += 1
                            stats[exp]["splits"][split]["neg"] += 1
                            
                        stats[exp]["total"] += 1
                        stats[exp]["splits"][split]["total"] += 1
                        
                        # Extract and save
                        crop_img = img[ny1:ny2, nx1:nx2]
                        crop_img_name = f"{base_name}_p{p['id']}.jpg"
                        exp_dir = os.path.join(output_dir_base, f"stage2_vest_crops_{exp}")
                        crop_img_path = os.path.join(exp_dir, split, "images", crop_img_name)
                        cv2.imwrite(crop_img_path, crop_img)
                        
                        crop_lbl_path = os.path.join(exp_dir, split, "labels", f"{base_name}_p{p['id']}.txt")
                        valid_label = True
                        with open(crop_lbl_path, "w") as f_lbl:
                            for v in p['vests']:
                                vx1, vy1, vx2, vy2 = v['box']
                                cvx1 = max(0, vx1 - nx1)
                                cvy1 = max(0, vy1 - ny1)
                                cvx2 = min(crop_w, vx2 - nx1)
                                cvy2 = min(crop_h, vy2 - ny1)
                                
                                cvw = cvx2 - cvx1
                                cvh = cvy2 - cvy1
                                
                                if cvw > 0 and cvh > 0:
                                    ccx = (cvx1 + cvw/2) / crop_w
                                    ccy = (cvy1 + cvh/2) / crop_h
                                    nw = cvw / crop_w
                                    nh = cvh / crop_h
                                    
                                    # Validation
                                    if not (0 <= ccx <= 1 and 0 <= ccy <= 1 and 0 < nw <= 1 and 0 < nh <= 1):
                                        validation_errors.append(f"Invalid bounds: {crop_lbl_path}")
                                        valid_label = False
                                    else:
                                        f_lbl.write(f"0 {ccx:.6f} {ccy:.6f} {nw:.6f} {nh:.6f}\n")
                        
                        # Visual Validation
                        if img_name in vis_samples[split]:
                            vis_dir = os.path.join(audit_out_dir, f"vis_exp_{exp}", split)
                            os.makedirs(vis_dir, exist_ok=True)
                            vis_img = crop_img.copy()
                            for v in p['vests']:
                                vx1, vy1, vx2, vy2 = v['box']
                                cvx1 = int(max(0, vx1 - nx1))
                                cvy1 = int(max(0, vy1 - ny1))
                                cvx2 = int(min(crop_w, vx2 - nx1))
                                cvy2 = int(min(crop_h, vy2 - ny1))
                                if cvx2 > cvx1 and cvy2 > cvy1:
                                    cv2.rectangle(vis_img, (cvx1, cvy1), (cvx2, cvy2), (255, 0, 0), 2)
                                    cv2.putText(vis_img, "Vest", (cvx1, max(10, cvy1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
                            cv2.imwrite(os.path.join(vis_dir, f"{p['status']}_{crop_img_name}"), vis_img)

    unassociated_csv.close()
    ambiguous_csv.close()
    
    with open(os.path.join(audit_out_dir, "crop_generation_stats.json"), "w") as f:
        json.dump({
            "expansions": stats,
            "global": global_stats,
            "validation_errors": len(validation_errors)
        }, f, indent=2)
        
    print("Generation complete!")

if __name__ == "__main__":
    main()
