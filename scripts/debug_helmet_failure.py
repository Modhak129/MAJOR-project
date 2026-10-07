import os
import cv2
import sys
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def main():
    img_path = r"C:\Users\Vishnu\.gemini\antigravity\brain\3a395107-a56b-4df5-a7db-327a4da905c5\.user_uploaded\media_1791024184799.png"
    out_dir = Path(r"C:\Major Project\Project-2\reports\two_stage_debug")
    
    # 1. Load image and crop out UI to get raw frame
    full_img = cv2.imread(img_path)
    # The image is a screenshot of a web UI. The video frame is roughly the top 65%.
    # We will just use the whole image or crop it manually. Let's crop top 60%
    H, W = full_img.shape[:2]
    frame = full_img[0:int(H*0.65), :]
    cv2.imwrite(str(out_dir / "problematic_frame.jpg"), frame)
    
    with open(out_dir / "runtime_trace.txt", "w") as trace:
        trace.write("PHASE 1 — TRACE THE ACTUAL RUNTIME\n")
        
        # Load models
        stage1_pt = r"C:\Major Project\Project-2\runs\detect\edge\weights\training\clean_sh17_yolov8n_640\weights\best.pt"
        stage2_pt = r"C:\Major Project\Project-2\runs\detect\edge\weights\training\hardhat_helmet_yolov8n_640\weights\best.pt"
        
        model1 = YOLO(stage1_pt)
        model2 = YOLO(stage2_pt)
        
        # Run Stage 1 (we won't use tracker for a single frame, just predict to get the person box)
        res1 = model1.predict(frame, conf=0.25, imgsz=320, verbose=False)[0]
        
        persons = []
        vests = []
        gloves = []
        helmets_s1 = []
        
        for box in res1.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            obj = {"cls": cls, "conf": conf, "box": (x1, y1, x2, y2)}
            
            if cls == 0: persons.append(obj)
            elif cls == 12: vests.append(obj)
            elif cls == 9: gloves.append(obj)
            elif cls == 14: helmets_s1.append(obj)
            
        trace.write(f"Stage 1 detections: {len(res1.boxes)} objects\n")
        trace.write(f"Persons: {len(persons)}, Vests: {len(vests)}, Gloves: {len(gloves)}, SH17 Helmets: {len(helmets_s1)}\n")
        
        # Let's find the main person (Worker 1 in screenshot)
        # It's the large box on the left.
        if not persons:
            trace.write("ERROR: No person detected by Stage 1!\n")
            return
            
        # Sort by area to get the biggest person
        persons.sort(key=lambda p: (p['box'][2]-p['box'][0])*(p['box'][3]-p['box'][1]), reverse=True)
        main_person = persons[0]
        px1, py1, px2, py2 = main_person['box']
        pw, ph = px2 - px1, py2 - py1
        
        trace.write(f"\nPHASE 2 — CHECK THE PERSON BOUNDING BOX\n")
        trace.write(f"track_id: 1\n")
        trace.write(f"person_x1: {px1:.2f}\nperson_y1: {py1:.2f}\nperson_x2: {px2:.2f}\nperson_y2: {py2:.2f}\n")
        trace.write(f"person_width: {pw:.2f}\nperson_height: {ph:.2f}\n")
        
        # Draw Stage 1 box
        img_s1 = frame.copy()
        cv2.rectangle(img_s1, (int(px1), int(py1)), (int(px2), int(py2)), (0, 255, 0), 2)
        cv2.imwrite(str(out_dir / "stage1_person_box.jpg"), img_s1)
        
        # 20% Crop
        exp = 0.20
        ex1 = max(0, int(px1 - pw * exp / 2))
        ey1 = max(0, int(py1 - ph * exp / 2))
        ex2 = min(frame.shape[1], int(px2 + pw * exp / 2))
        ey2 = min(frame.shape[0], int(py2 + ph * exp / 2))
        
        trace.write(f"\nexpanded_x1: {ex1}\nexpanded_y1: {ey1}\nexpanded_x2: {ex2}\nexpanded_y2: {ey2}\n")
        
        # PHASE 3 - Save exact crop
        crop = frame[ey1:ey2, ex1:ex2]
        cv2.imwrite(str(out_dir / "crops" / "frame_0000_track_1_original.jpg"), frame[int(py1):int(py2), int(px1):int(px2)])
        cv2.imwrite(str(out_dir / "crops" / "frame_0000_track_1_expanded.jpg"), crop)
        cv2.imwrite(str(out_dir / "expanded_crop.jpg"), crop)
        
        # PHASE 4 - Run Stage 2 on exact crop
        trace.write(f"\nPHASE 4 — RUN STAGE 2 ON EXACT CROP\n")
        res2_raw = model2.predict(crop, conf=0.01, verbose=False)[0]
        
        for thresh in [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.45, 0.50]:
            dets = [float(b.conf[0]) for b in res2_raw.boxes if float(b.conf[0]) >= thresh and int(b.cls[0]) == 0]
            max_c = max(dets) if dets else 0.0
            trace.write(f"threshold {thresh}: count={len(dets)}, max_conf={max_c:.4f}, all={dets}\n")
            
        # PHASE 5 - Test Crop Expansion
        trace.write(f"\nPHASE 5 — TEST CROP EXPANSION\n")
        trace.write("expansion | crop dimensions | helmet detected? | max confidence | helmet bbox\n")
        with open(out_dir / "crop_expansion_sweep.csv", "w") as f:
            f.write("expansion,crop_w,crop_h,detected,max_conf\n")
            for e in [0.00, 0.10, 0.20, 0.30, 0.40, 0.50]:
                tx1 = max(0, int(px1 - pw * e / 2))
                ty1 = max(0, int(py1 - ph * e / 2))
                tx2 = min(frame.shape[1], int(px2 + pw * e / 2))
                ty2 = min(frame.shape[0], int(py2 + ph * e / 2))
                tcrop = frame[ty1:ty2, tx1:tx2]
                tres = model2.predict(tcrop, conf=0.01, verbose=False)[0]
                tdets = [float(b.conf[0]) for b in tres.boxes if int(b.cls[0]) == 0]
                tmax = max(tdets) if tdets else 0.0
                t_det = "YES" if tmax >= 0.40 else "NO"
                trace.write(f"{e:.2f} | {tx2-tx1}x{ty2-ty1} | {t_det} | {tmax:.4f} | {tdets}\n")
                f.write(f"{e:.2f},{tx2-tx1},{ty2-ty1},{t_det},{tmax:.4f}\n")
                
        # PHASE 6 - Test Input Resolution
        trace.write(f"\nPHASE 6 — TEST INPUT RESOLUTION (on 20% crop)\n")
        with open(out_dir / "resolution_sweep.csv", "w") as f:
            f.write("imgsz,detected,confidence\n")
            for sz in [320, 640, 960]:
                sres = model2.predict(crop, conf=0.01, imgsz=sz, verbose=False)[0]
                sdets = [float(b.conf[0]) for b in sres.boxes if int(b.cls[0]) == 0]
                smax = max(sdets) if sdets else 0.0
                trace.write(f"imgsz={sz}: detected={'YES' if smax>=0.4 else 'NO'}, conf={smax:.4f}\n")
                f.write(f"{sz},{'YES' if smax>=0.4 else 'NO'},{smax:.4f}\n")
                
        # PHASE 8 & 9 - Coordinate Transformation
        trace.write(f"\nPHASE 8 & 9 — COORDINATE TRANSFORMATION\n")
        res2_prod = model2.predict(crop, conf=0.40, verbose=False)[0]
        img_s2_full = frame.copy()
        
        if len(res2_prod.boxes) > 0:
            h_box = res2_prod.boxes[0]
            hx1, hy1, hx2, hy2 = map(float, h_box.xyxy[0])
            hconf = float(h_box.conf[0])
            trace.write(f"Crop bbox: {hx1:.1f}, {hy1:.1f}, {hx2:.1f}, {hy2:.1f} (conf {hconf:.2f})\n")
            
            fx1, fy1, fx2, fy2 = int(hx1 + ex1), int(hy1 + ey1), int(hx2 + ex1), int(hy2 + ey1)
            trace.write(f"Full-frame bbox: {fx1}, {fy1}, {fx2}, {fy2}\n")
            
            cv2.rectangle(img_s2_full, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)
            cv2.putText(img_s2_full, f"Helmet {hconf:.2f}", (fx1, fy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        else:
            trace.write("No helmet detected at production threshold (0.40) on 20% crop.\n")
            
        cv2.imwrite(str(out_dir / "stage2_result_fullframe.jpg"), img_s2_full)
        
        # PHASE 15 - VEST FAILURE
        trace.write(f"\nPHASE 15 — VEST FAILURE\n")
        trace.write(f"Vests detected in full frame by Stage 1: {len(vests)}\n")
        for v in vests:
            trace.write(f"Vest: conf={v['conf']:.2f}, bbox={v['box']}\n")
            # Check overlap
            vx1, vy1, vx2, vy2 = v['box']
            inter_x1 = max(px1, vx1)
            inter_y1 = max(py1, vy1)
            inter_x2 = min(px2, vx2)
            inter_y2 = min(py2, vy2)
            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                item_area = (vx2 - vx1) * (vy2 - vy1)
                iou = inter_area / item_area
                trace.write(f"Overlap with person: {iou*100:.1f}%\n")
                if iou > 0.5:
                    trace.write("Vest successfully associated with person!\n")
                else:
                    trace.write("Vest NOT associated (overlap <= 50%).\n")

        # PHASE 16 - CONTROL EXPERIMENT
        trace.write(f"\nPHASE 16 — CONTROL EXPERIMENT\n")
        # A) SH17 on full frame
        trace.write(f"A) Stage-1 SH17 Full Frame Helmets: {[b['conf'] for b in helmets_s1]}\n")
        # B) Stage-2 HardHat on full frame
        res_b = model2.predict(frame, conf=0.01, verbose=False)[0]
        trace.write(f"B) Stage-2 HardHat Full Frame Helmets (conf>0.4): {[float(b.conf[0]) for b in res_b.boxes if int(b.cls[0])==0 and float(b.conf[0])>0.4]}\n")
        # C) Stage-2 on 20% crop
        trace.write(f"C) Stage-2 20% crop: {[float(b.conf[0]) for b in res2_prod.boxes if int(b.cls[0])==0]}\n")
        # D) Stage-2 on 40% crop
        tx1 = max(0, int(px1 - pw * 0.40 / 2))
        ty1 = max(0, int(py1 - ph * 0.40 / 2))
        tx2 = min(frame.shape[1], int(px2 + pw * 0.40 / 2))
        ty2 = min(frame.shape[0], int(py2 + ph * 0.40 / 2))
        tres40 = model2.predict(frame[ty1:ty2, tx1:tx2], conf=0.40, verbose=False)[0]
        trace.write(f"D) Stage-2 40% crop: {[float(b.conf[0]) for b in tres40.boxes if int(b.cls[0])==0]}\n")

if __name__ == "__main__":
    main()
