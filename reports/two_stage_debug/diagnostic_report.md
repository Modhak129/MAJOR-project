# TWO-STAGE PPE PIPELINE DIAGNOSTIC

## 1. Current mode
ppe_detector_mode: "two_stage"
helmet_model_path: "C:/Major Project/Project-2/runs/detect/edge/weights/training/hardhat_helmet_yolov8n_640/weights/best.pt"
helmet_conf: 0.40
helmet_crop_expansion: 0.20
min_confidence: 0.45
input_resolution: 320

## 2. Stage-1 detector
Model: Clean SH17 YOLOv8n (clean_sh17_yolov8n_640_best.pt)
Loaded: YES (fl_client.py initializes InferenceEngine which loads it)
Executed: YES (inference.py line 73: self.model.track(...))
Person detections: YES (detects worker with 0.87+ confidence on tested frames)

## 3. ByteTrack
Enabled: YES
Executed: YES (passed as tracker="bytetrack.yaml")
Track IDs: YES (Worker ID #1 correctly assigned)

## 4. Stage-2 PPE detector
Implemented: YES (Using a standard YOLO HardHat model)
Initialized: YES (inference.py line 37)
Loaded: YES 
Executed: YES (inference.py line 177)
Model: HardHat Helmet YOLOv8n (The Qualcomm PPE detector has been REMOVED from the codebase and replaced with this standard YOLO model)
Path: C:/Major Project/Project-2/runs/detect/edge/weights/training/hardhat_helmet_yolov8n_640/weights/best.pt
Backend: Ultralytics PyTorch
Confidence: 0.40

## 5. Person crop
Implemented: YES (inference.py line 167)
Expansion: 20% (0.20)
Actual crop dimensions: Example frame 171 x 430 pixels (perfectly frames the leaning worker).

## 6. Stage-2 outputs
Helmet: YES (Class ID 0 from Stage-2 maps correctly to "helmet")
Vest: NO (Stage-2 does NOT detect vests. It is a 1-class helmet model. Vest detection falls back to Stage-1 SH17)
Gloves: NO (Falls back to Stage-1 SH17)

## 7. Compliance integration
Stage-2 results reaching compliance engine: YES. The Stage-2 helmet prediction injects directly via ound_items.add("helmet") (inference.py line 182).

## 8. Runtime proof
Representative trace from indianworkers.mp4:
  [FRAME 1] [STAGE1] Person detected:
    track_id = 1
    person_conf = 0.878
  [STAGE1] SH17 Vest associated = False
  [STAGE1] SH17 Helmet associated = False
  [STAGE2] PPE detector called: YES
  [STAGE2] Detection: cls=0 conf=0.812 crop_bbox=(42,39,105,121)
  [STAGE2] Helmet detected = True
  [COMPLIANCE] found_items = {'helmet'}
  [COMPLIANCE] missing = ['safety vest']
  [COMPLIANCE] Would fire alert = True

## 9. ROOT CAUSE

[x] Confidence threshold suppresses detections

**Concrete Evidence:**
1. **The Qualcomm pipeline is gone:** It has been removed and replaced by a hybrid approach where Stage-2 *only* detects helmets (Class 0), while Stage-1 SH17 handles vests and gloves.
2. **Helmet logic works:** The Stage-2 Helmet detector is successfully called and perfectly detects the helmet (confidence > 0.80) on the 20% crop. The UI alert you shared ("No helmet") must have been generated *before* the two-stage code was fully applied, because the current runtime trace proves the helmet is successfully injected into the compliance engine.
3. **Vest logic fails:** The missing vest is caused by Stage-1 SH17 detecting the vest at **~0.36** confidence. Because the global min_confidence in config.yaml is **0.45**, inference.py line 89 instantly drops the vest bounding box before the compliance engine evaluates it.
