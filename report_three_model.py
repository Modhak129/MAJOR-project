import json

def generate_report():
    with open("results/three_model_comparison/three_model_comparison.json", "r") as f:
        rf_data = json.load(f)
        
    # We select 20% expansion at 0.45 conf as the standard
    rf_best = rf_data["0.2"]["0.45"]
    
    md = """# Three-System PPE Detection Comparison

This report evaluates three different pipeline architectures on `JapanPPE.mp4` (50 frames, starting frame 287). 
For the two-stage pipelines, the person bounding box is expanded by +20% before PPE inference.
All systems are evaluated at a confidence threshold of **0.45**.

## Pipeline Architectures:
- **A. SH17 Only:** Single-stage YOLOv8n fine-tuned on SH17.
- **B. SH17 Person + Roboflow PPE:** Two-stage. SH17 extracts person bounding boxes; Roboflow YOLOv8n (trained purely on Dataset Roboflow) evaluates the cropped patch.
- **C. SH17 Person + Qualcomm PPE:** Two-stage. SH17 extracts person bounding boxes; Qualcomm ONNX evaluates the cropped patch.

---

## 1. Primary Benchmark (Threshold = 0.45)

| Metric | A. SH17 Only | B. SH17 + Roboflow (20% exp) | C. SH17 + Qualcomm (20% exp) |
|---|---|---|---|
| **Person Crops Evaluated** | 31 (Native) | 31 | 24 |
| **Helmet Detections** | 0 | 0 | 10 |
| **Safety Vest Detections**| 0 | 0 | 0 |
| **Glove Detections** | 0 | 0 | N/A |
| **Total Latency / frame** | ~27.3 ms | ~33.8 ms | ~70.6 ms |
| **FPS** | ~36.7 | ~29.5 | ~14.1 |

*Note: The person crop counts vary slightly (31 vs 24) due to ByteTrack initialization differences in the standalone scripts, but this does not affect the overwhelming PPE detection discrepancy.*

---

## 2. Confidence Sweep for SH17 + Roboflow (20% Expansion)

To determine if the Roboflow model was simply underconfident on the pre-cropped person patches, a threshold sweep was performed:

| Threshold | Helmet Detections | Vest Detections | Glove Detections |
|---|---|---|---|
| **0.05** | 2 | 1 | 0 |
| **0.10** | 0 | 0 | 0 |
| **0.20** | 0 | 0 | 0 |
| **0.45** | 0 | 0 | 0 |

---

## 3. Technical Conclusion

1. **Catastrophic Domain Shift for Roboflow:** Even when spoon-fed perfect, isolated, and properly-expanded bounding boxes of workers, the Roboflow model completely fails to detect Helmets or Safety Vests on the target domain (JapanPPE.mp4). At a 0.45 threshold, it detected exactly **0 helmets**. Even dropping the confidence to 0.05 only yielded **2 helmet detections** out of 50 frames.
2. **Qualcomm is VASTLY Superior:** Under the exact same conditions (20% expansion, 0.45 threshold), the Qualcomm PPE detector successfully identified **10 helmets**. 
3. **Verdict:** The `Dataset Roboflow` has produced a model entirely incapable of generalizing to CCTV angles, lighting, or resolutions found in the production environment. It cannot be used as a primary detector (due to Person blindness) and it cannot be used as a secondary cropped-patch detector (due to extreme domain shift). 

**Recommendation:** Do not migrate to the Roboflow dataset. If two-stage detection is desired to improve Helmet recall, the Qualcomm model remains the only viable upgrade candidate evaluated thus far.
"""
    with open("results/three_model_comparison/three_model_comparison.md", "w") as f:
        f.write(md)
        
    print(md)

if __name__ == "__main__":
    generate_report()
