"""
RUNTIME DIAGNOSTIC: Two-Stage PPE Pipeline Verification
This script simulates the EXACT startup path used by fl_client.py,
then processes real video frames and logs every step of the pipeline.
"""
import sys
import os
import yaml
import cv2
import time

sys.path.insert(0, r"C:\Major Project\Project-2\edge")
sys.path.insert(0, r"C:\Major Project\Project-2")

os.chdir(r"C:\Major Project\Project-2")

from inference import InferenceEngine

OUT_DIR = r"C:\Major Project\Project-2\reports\two_stage_debug"
os.makedirs(OUT_DIR, exist_ok=True)

log_lines = []
def log(msg):
    print(msg, flush=True)
    log_lines.append(msg)

def main():
    # =============================================
    # PHASE 1: Reproduce exact startup path
    # =============================================
    log("=" * 70)
    log("PHASE 1: REPRODUCING EXACT STARTUP PATH FROM fl_client.py")
    log("=" * 70)

    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # fl_client.py line 365: tier selection
    vram_mb = 2048  # default for node_constrained
    tier = "nano" if vram_mb < config['adaptive']['vram_threshold_mb'] else "small"
    log(f"Tier selected: {tier}")

    # fl_client.py line 369: model_path = config['models'][tier]
    model_path = config['models'][tier]
    log(f"Stage-1 model_path from config['models']['{tier}']: {model_path}")
    log(f"Stage-1 model file exists: {os.path.exists(model_path)}")

    # fl_client.py line 384: inference_engine = InferenceEngine(config, model_path, client_id, dashboard_url)
    log(f"\nInitializing InferenceEngine(config, '{model_path}', 'node_constrained', 'http://localhost:5000')...")
    engine = InferenceEngine(config, model_path, "node_constrained", "http://localhost:5000")

    log(f"\n--- InferenceEngine state after __init__ ---")
    log(f"ppe_detector_mode = '{engine.ppe_detector_mode}'")
    log(f"helmet_conf = {engine.helmet_conf}")
    log(f"helmet_crop_expansion = {engine.helmet_crop_expansion}")
    log(f"min_conf (global Stage-1 threshold) = {engine.min_conf}")
    log(f"helmet_model is None = {engine.helmet_model is None}")
    log(f"helmet_model type = {type(engine.helmet_model)}")

    if engine.helmet_model is not None:
        log(f"Stage-2 model names: {engine.helmet_model.names}")
    else:
        log("*** CRITICAL: Stage-2 helmet model was NOT loaded! ***")

    log(f"Stage-1 model names: {engine.model.names}")

    # =============================================
    # PHASE 2: Process real video frames with instrumented logging
    # =============================================
    log("\n" + "=" * 70)
    log("PHASE 2: RUNTIME FRAME PROCESSING WITH INSTRUMENTED LOGGING")
    log("=" * 70)

    videos = [
        r"data\raw_dataset\source_files\source_files\hardhat.mp4",
        r"data\raw_dataset\source_files\source_files\indianworkers.mp4",
    ]

    for vid_path in videos:
        if not os.path.exists(vid_path):
            log(f"Video not found: {vid_path}")
            continue

        log(f"\n--- Processing: {vid_path} ---")
        cap = cv2.VideoCapture(vid_path)
        engine.reset_tracker()

        frame_count = 0
        worker_frames_logged = 0

        while frame_count < 50:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            H, W = frame.shape[:2]

            # Replicate process_frame logic step by step with logging
            imgsz_val = config['inference'].get('input_resolution', 320)
            results = engine.model.track(frame, persist=True, tracker="bytetrack.yaml",
                                          verbose=False, imgsz=imgsz_val, device='0')[0]

            names = engine.model.names
            PERSON_CLASS_ID = next((k for k, v in names.items() if v.lower() == "person"), 0)
            HELMET_CLASS_ID = next((k for k, v in names.items() if v.lower() == "helmet"), 14)
            VEST_CLASS_ID = next((k for k, v in names.items() if v.lower() == "safety-vest"), 12)

            persons = []
            items = []

            for box in results.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                worker_id = int(box.id[0]) if box.id is not None else None

                if conf < engine.min_conf:
                    if cls == VEST_CLASS_ID and worker_frames_logged < 5:
                        log(f"  [FRAME {frame_count}] VEST SUPPRESSED by min_conf: cls={cls} conf={conf:.3f} < {engine.min_conf}")
                    if cls == HELMET_CLASS_ID and worker_frames_logged < 5:
                        log(f"  [FRAME {frame_count}] SH17 HELMET SUPPRESSED by min_conf: cls={cls} conf={conf:.3f} < {engine.min_conf}")
                    continue

                x1, y1, x2, y2 = map(float, box.xyxy[0])
                obj = {"cls": cls, "id": worker_id, "box": (x1, y1, x2, y2), "conf": conf}

                if cls == PERSON_CLASS_ID:
                    persons.append(obj)
                else:
                    items.append(obj)

            if not persons:
                continue

            # Log detail for first few workers
            for person in persons:
                if worker_frames_logged >= 10:
                    break

                px1, py1, px2, py2 = person["box"]
                bw, bh = px2 - px1, py2 - py1

                log(f"\n  [FRAME {frame_count}] [STAGE1] Person detected:")
                log(f"    track_id = {person['id']}")
                log(f"    person_bbox = ({px1:.1f}, {py1:.1f}, {px2:.1f}, {py2:.1f})")
                log(f"    person_conf = {person['conf']:.3f}")
                log(f"    person_size = {bw:.1f} x {bh:.1f}")

                # Check SH17 vests/helmets for this person
                sh17_vest_found = False
                sh17_helmet_found = False
                for item in items:
                    ix1, iy1, ix2, iy2 = item["box"]
                    inter_x1 = max(px1, ix1)
                    inter_y1 = max(py1, iy1)
                    inter_x2 = min(px2, ix2)
                    inter_y2 = min(py2, iy2)
                    overlap = 0
                    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                        item_area = (ix2 - ix1) * (iy2 - iy1)
                        overlap = inter_area / item_area if item_area > 0 else 0

                    if item["cls"] == VEST_CLASS_ID:
                        log(f"    [STAGE1] Vest item: conf={item['conf']:.3f}, overlap={overlap:.2f}, bbox=({ix1:.0f},{iy1:.0f},{ix2:.0f},{iy2:.0f})")
                        if overlap > 0.5:
                            sh17_vest_found = True
                    if item["cls"] == HELMET_CLASS_ID:
                        log(f"    [STAGE1] Helmet item: conf={item['conf']:.3f}, overlap={overlap:.2f}")
                        if overlap > 0.5:
                            sh17_helmet_found = True

                log(f"    [STAGE1] SH17 Vest associated = {sh17_vest_found}")
                log(f"    [STAGE1] SH17 Helmet associated = {sh17_helmet_found}")

                # TWO-STAGE HELMET DETECTION
                if engine.ppe_detector_mode == "two_stage" and engine.helmet_model is not None:
                    exp = engine.helmet_crop_expansion
                    ex1 = max(0, int(px1 - (bw * exp / 2)))
                    ey1 = max(0, int(py1 - (bh * exp / 2)))
                    ex2 = min(W, int(px2 + (bw * exp / 2)))
                    ey2 = min(H, int(py2 + (bh * exp / 2)))

                    log(f"    [STAGE2] Crop expansion = {exp}")
                    log(f"    [STAGE2] Expanded crop bbox = ({ex1}, {ey1}, {ex2}, {ey2})")
                    log(f"    [STAGE2] Crop size = {ex2-ex1} x {ey2-ey1}")

                    crop = frame[ey1:ey2, ex1:ex2]
                    log(f"    [STAGE2] Crop extracted: shape={crop.shape}")

                    # Save crop for first couple
                    if worker_frames_logged < 3:
                        crop_path = os.path.join(OUT_DIR, "crops", f"runtime_frame{frame_count}_track{person['id']}.jpg")
                        os.makedirs(os.path.dirname(crop_path), exist_ok=True)
                        cv2.imwrite(crop_path, crop)
                        log(f"    [STAGE2] Crop saved: {crop_path}")

                    # Run Stage-2 at production threshold
                    h_res = engine.helmet_model.predict(crop, conf=engine.helmet_conf, verbose=False, device='0')[0]
                    log(f"    [STAGE2] PPE detector called: YES")
                    log(f"    [STAGE2] Production threshold = {engine.helmet_conf}")
                    log(f"    [STAGE2] Raw box count = {len(h_res.boxes) if h_res.boxes is not None else 0}")

                    helmet_found = False
                    if h_res.boxes is not None:
                        for h_box in h_res.boxes:
                            hcls = int(h_box.cls[0])
                            hconf = float(h_box.conf[0])
                            hx1, hy1, hx2, hy2 = map(float, h_box.xyxy[0])
                            log(f"    [STAGE2] Detection: cls={hcls} conf={hconf:.3f} crop_bbox=({hx1:.0f},{hy1:.0f},{hx2:.0f},{hy2:.0f})")
                            fx1 = int(hx1 + ex1)
                            fy1 = int(hy1 + ey1)
                            fx2 = int(hx2 + ex1)
                            fy2 = int(hy2 + ey1)
                            log(f"    [STAGE2] Full-frame bbox = ({fx1},{fy1},{fx2},{fy2})")
                            if hcls == 0:
                                helmet_found = True
                    else:
                        log(f"    [STAGE2] h_res.boxes is None")

                    log(f"    [STAGE2] Helmet detected = {helmet_found}")

                    # Also run at ultra-low conf for diagnostic
                    h_res_diag = engine.helmet_model.predict(crop, conf=0.01, verbose=False, device='0')[0]
                    diag_confs = []
                    if h_res_diag.boxes is not None:
                        for hb in h_res_diag.boxes:
                            if int(hb.cls[0]) == 0:
                                diag_confs.append(float(hb.conf[0]))
                    log(f"    [STAGE2] Diagnostic (conf=0.01): helmet confs = {diag_confs}")

                else:
                    log(f"    [STAGE2] PPE detector called: NO (mode={engine.ppe_detector_mode}, model_loaded={engine.helmet_model is not None})")

                # COMPLIANCE
                found_items = set()
                if sh17_vest_found:
                    found_items.add("safety_vest")
                if engine.ppe_detector_mode == "two_stage" and engine.helmet_model is not None:
                    # helmet_found was set above
                    if helmet_found:
                        found_items.add("helmet")
                elif sh17_helmet_found:
                    found_items.add("helmet")

                missing = []
                if "helmet" not in found_items:
                    missing.append("helmet")
                if "safety_vest" not in found_items:
                    missing.append("safety vest")

                log(f"    [COMPLIANCE] found_items = {found_items}")
                log(f"    [COMPLIANCE] missing = {missing}")
                log(f"    [COMPLIANCE] Would fire alert = {len(missing) > 0}")

                worker_frames_logged += 1

        cap.release()
        log(f"Processed {frame_count} frames from {vid_path}")

    # Write log
    with open(os.path.join(OUT_DIR, "runtime_diagnostic.txt"), "w") as f:
        f.write("\n".join(log_lines))

    log(f"\nDiagnostic saved to {OUT_DIR}/runtime_diagnostic.txt")

if __name__ == "__main__":
    main()
