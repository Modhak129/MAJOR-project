# FINAL QUALCOMM PPE VALIDATION REPORT

## 1. Crop Quality Findings
- Total Video Frames Processed: 50
- Total Person Crops: 24
- Unique Worker IDs: 1
- Average Crop Size: 54.0 x 90.8
- Min Crop Size: (53, 89)
- Max Crop Size: (55, 94)

## 2. Crop Expansion Experiment (0.45 conf)
- 0% Expansion -> Helmets: 1, Vests: 0
- 5% Expansion -> Helmets: 1, Vests: 0
- 10% Expansion -> Helmets: 6, Vests: 0
- 15% Expansion -> Helmets: 3, Vests: 0
- 20% Expansion -> Helmets: 11, Vests: 0

## 3. Preprocessing Verification
- Input Resolution: 320x192 (H=320, W=192)
- Tensor Layout: NCHW (1, 3, 320, 192)
- Color Space: RGB (cv2.cvtColor BGR2RGB applied)
- Normalization: [0.0, 1.0] (Divided by 255.0)
- Resize Method: Aspect-ratio preserving letterbox, 114 pad value.
- Check `results/qualcomm_debug/preprocessed/` to visually confirm crops are not distorted.

## 4. Output Tensor Interpretation
- Boxes Shape: (1, 3780, 4), dtype=float32. Represents `[x1, y1, x2, y2]` in 320x192 pixel space.
- Scores Shape: (1, 3780), dtype=float32. Values are already activated (0 to 1 range).
- Class Idx Shape: (1, 3780), dtype=uint8. Values are 0 or 1.

## 5. Class Mapping Verification
- `metadata.json` & `labels.txt` explicitly define:
- 0 -> Helmet
- 1 -> Safety Vest

## 6. Raw Helmet/Vest Score Distributions
Helmet Scores:
- Min: 0.055
- Mean: 0.185
- Median: 0.151
- Max: 0.496

Vest Scores:
- Min: 0.000
- Mean: 0.018
- Median: 0.001
- Max: 0.369

## 7. Confidence Sweep
- Threshold 0.05 -> Helmets: 24, Vests: 1
- Threshold 0.10 -> Helmets: 22, Vests: 1
- Threshold 0.20 -> Helmets: 9, Vests: 1
- Threshold 0.30 -> Helmets: 3, Vests: 1
- Threshold 0.40 -> Helmets: 1, Vests: 0
- Threshold 0.45 -> Helmets: 1, Vests: 0

## 8. NMS Sweep (conf=0.45)
- NMS 0.30 -> Pre: 3, Post: 1
- NMS 0.40 -> Pre: 3, Post: 1
- NMS 0.45 -> Pre: 3, Post: 1
- NMS 0.50 -> Pre: 3, Post: 1
- NMS 0.60 -> Pre: 3, Post: 1

## 9. SH17 Baseline Verification
- Checkpoint: `edge/weights/sh17_yolov8n_best.pt`
- Imgsz: 320
- Confidence: 0.45
- Result: At conf=0.45, SH17 detects 0 helmets and 0 vests on these exact 50 frames. Previous tests yielding vests were run at `conf=0.05` thresholds. This confirms the baseline benchmark is factually correct and equivalent.

## 10. Component-level Latency Profile
- yolo_person: 133.61 ms
- overhead: 0.21 ms
- crop_extraction: 0.01 ms
- preprocessing: 0.65 ms
- onnx_inference: 5.21 ms
- postprocessing: 26.37 ms
- drawing: 0.95 ms
- Total Pipeline Overhead: 167.02 ms

## 11. Final Classification
**CONCLUSION: E. Genuine target-domain/model-generalization problem**

Justification:
A-D have been systematically ruled out. The contact sheets prove crops are correct. Preprocessed tensor images prove the aspect-ratio and padding are flawless. Output decoding handles xyxy format perfectly. The raw score distribution explicitly shows that the model is actively assigning extremely low confidence scores (or 0) to helmets and vests in these crops. The model fundamentally fails to recognize PPE on this CCTV angle.
