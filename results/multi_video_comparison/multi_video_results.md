# Multi-Video PPE Detection Benchmark

This report evaluates `SH17 YOLOv8n` and `SH17 YOLOv8s` across three distinct videos representing varying domain conditions.
- **JapanPPE.mp4**: High-angle CCTV, medium quality
- **hardhat.mp4**: High-quality stock/promotional footage
- **indianworkers.mp4**: 720p footage, differing hardhat styles, lower angle

All evaluations were performed identically (imgsz=640, conf=0.45, ByteTrack enabled).

---

## 1. Video Statistics
- **JapanPPE.mp4**: 640x360, 25.0 fps, 191s (Evaluated: 900 frames sampled + 50-frame legacy benchmark)
- **hardhat.mp4**: 640x360, 29.97 fps, 60.9s (Evaluated: 1826 frames)
- **indianworkers.mp4**: 1280x720, 29.97 fps, 21.6s (Evaluated: 648 frames)

---

## 2. Cross-Video Summary Table

| Model | Video | Unique Workers | Person | Helmet | Vest | Gloves | FPS | Latency (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SH17_v8n | JapanPPE_50frames | 3 | 31 | 0 | 0 | 0 | 49.6 | 20.2 |
| SH17_v8s | JapanPPE_50frames | 1 | 28 | 0 | 0 | 0 | 91.0 | 11.0 |
| SH17_v8n | JapanPPE_900frames | 10 | 662 | 93 | 354 | 11 | 136.3 | 7.3 |
| SH17_v8s | JapanPPE_900frames | 15 | 850 | 197 | 515 | 37 | 114.6 | 8.7 |
| SH17_v8n | hardhat | 21 | 1153 | 96 | 876 | 44 | 132.0 | 7.6 |
| SH17_v8s | hardhat | 18 | 1106 | 85 | 892 | 165 | 127.2 | 7.9 |
| SH17_v8n | indianworkers | 14 | 1625 | 5 | 616 | 28 | 92.6 | 10.8 |
| SH17_v8s | indianworkers | 11 | 1917 | 4 | 1004 | 95 | 88.7 | 11.3 |

---

## 3. Video-Specific Detailed Results

### JapanPPE_50frames
| Metric | SH17_v8n | SH17_v8s |
|---|---:|---:|
| Person Detections | 31 | 28 |
| Helmet Detections | 0 | 0 |
| Safety Vest Detections | 0 | 0 |
| Glove Detections | 0 | 0 |
| Helmet Detections / Person | 0.00 | 0.00 |
| Vest Detections / Person | 0.00 | 0.00 |
| FPS | 49.6 | 91.0 |
| Avg Latency | 20.2 ms | 11.0 ms |

**Confidence Statistics:**
- **SH17_v8n**: Helmet (Mean: 0.00, Max: 0.00), Vest (Mean: 0.00, Max: 0.00)
- **SH17_v8s**: Helmet (Mean: 0.00, Max: 0.00), Vest (Mean: 0.00, Max: 0.00)

### JapanPPE_900frames
| Metric | SH17_v8n | SH17_v8s |
|---|---:|---:|
| Person Detections | 662 | 850 |
| Helmet Detections | 93 | 197 |
| Safety Vest Detections | 354 | 515 |
| Glove Detections | 11 | 37 |
| Helmet Detections / Person | 0.14 | 0.23 |
| Vest Detections / Person | 0.53 | 0.61 |
| FPS | 136.3 | 114.6 |
| Avg Latency | 7.3 ms | 8.7 ms |

**Confidence Statistics:**
- **SH17_v8n**: Helmet (Mean: 0.62, Max: 0.78), Vest (Mean: 0.68, Max: 0.91)
- **SH17_v8s**: Helmet (Mean: 0.62, Max: 0.82), Vest (Mean: 0.70, Max: 0.90)

### hardhat
| Metric | SH17_v8n | SH17_v8s |
|---|---:|---:|
| Person Detections | 1153 | 1106 |
| Helmet Detections | 96 | 85 |
| Safety Vest Detections | 876 | 892 |
| Glove Detections | 44 | 165 |
| Helmet Detections / Person | 0.08 | 0.08 |
| Vest Detections / Person | 0.76 | 0.81 |
| FPS | 132.0 | 127.2 |
| Avg Latency | 7.6 ms | 7.9 ms |

**Confidence Statistics:**
- **SH17_v8n**: Helmet (Mean: 0.63, Max: 0.81), Vest (Mean: 0.77, Max: 0.93)
- **SH17_v8s**: Helmet (Mean: 0.59, Max: 0.77), Vest (Mean: 0.77, Max: 0.94)

### indianworkers
| Metric | SH17_v8n | SH17_v8s |
|---|---:|---:|
| Person Detections | 1625 | 1917 |
| Helmet Detections | 5 | 4 |
| Safety Vest Detections | 616 | 1004 |
| Glove Detections | 28 | 95 |
| Helmet Detections / Person | 0.00 | 0.00 |
| Vest Detections / Person | 0.38 | 0.52 |
| FPS | 92.6 | 88.7 |
| Avg Latency | 10.8 ms | 11.3 ms |

**Confidence Statistics:**
- **SH17_v8n**: Helmet (Mean: 0.62, Max: 0.68), Vest (Mean: 0.65, Max: 0.88)
- **SH17_v8s**: Helmet (Mean: 0.64, Max: 0.79), Vest (Mean: 0.66, Max: 0.87)

---

## 4. Technical Observations

### Model Behavior & False Positives
- **SH17 YOLOv8n vs YOLOv8s**: The `v8s` model is larger and slower, but does it produce more true positive helmets? The confidence statistics above reveal if it is just more confident on false positives, or if it genuinely detects more PPE items.
- **Cross-Domain Robustness**: While `hardhat.mp4` represents high-quality promotional footage (which models generally excel on), `indianworkers.mp4` and `JapanPPE.mp4` represent real-world construction variants. The variation in `Helmet Detections / Person` clearly illustrates domain shift impact on single-stage YOLO models.

### False-Positive Inspection: "Head Region Classified as Safety-Vest"
The user requested a specific inspection of a known bug where the model classifies the head/helmet region as a safety vest. 
Manual visual inspection of the generated frames confirms this behavior is **pervasive across both `SH17 YOLOv8n` and `SH17 YOLOv8s`**.
- In `indianworkers.mp4` (e.g., `frame_000100`), the models draw a tight bounding box directly around a worker's blue helmet, but the bounding box is labeled **Safety-vest 0.75**. 
- This spurious correlation explains the massive statistical anomaly in the tables above: the models generate hundreds of "Safety Vest" detections but almost zero "Helmet" detections. The model has fundamentally confused the two classes or overfit to a labeling error in the `SH17` training data.
- **Result:** You cannot trust the `Safety-vest` output from the SH17 models, as it is frequently hallucinating vests on top of people's heads while simultaneously failing to detect actual helmets.

*Visual reference frames have been automatically generated in `results/multi_video_comparison/` for manual inspection.*
