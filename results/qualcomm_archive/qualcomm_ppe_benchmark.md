# Qualcomm PPE Benchmark

Model: Qualcomm PPE-Detection (gear_guard_net-onnx-float)
Input: JapanPPE.mp4 (50 consecutive frames starting at 287)

## Metrics

Metric | SH17 YOLOv8n | YOLO+Qualcomm
--- | --- | ---
Total Persons Tracked | 24 | 24
Helmet Detections | 0 | 3
Safety Vest Detections | 0 | 0
Pipeline Latency (per frame) | 10.7 ms | 62.1 ms
PPE Inference Only (per crop) | N/A | 4.0 ms
End-to-End FPS | 93.5 | 16.1
