# Qualcomm PPE Crop Expansion Benchmark

Model: Qualcomm PPE-Detection (gear_guard_net-onnx-float)
Input: JapanPPE.mp4 (50 consecutive frames starting at 287)

| Expansion | Unique Workers | Person Crops | Helmet Detections | Vest Detections | Helmet Rate | Vest Rate | Avg H-Conf | Avg V-Conf | PPE ms | Pipeline ms | FPS |
|-----------|----------------|--------------|-------------------|-----------------|-------------|-----------|------------|------------|--------|--------------|-----|
| 0% | 1 | 24 | 1 | 0 | 4.2% | 0.0% | 0.50 | 0.00 | 4.5 | 133.2 | 7.5 |
| 5% | 1 | 24 | 2 | 0 | 8.3% | 0.0% | 0.59 | 0.00 | 4.5 | 68.3 | 14.6 |
| 10% | 1 | 24 | 3 | 0 | 12.5% | 0.0% | 0.62 | 0.00 | 4.6 | 68.7 | 14.6 |
| 15% | 1 | 24 | 1 | 0 | 4.2% | 0.0% | 0.48 | 0.00 | 4.6 | 66.6 | 15.0 |
| 20% | 1 | 24 | 10 | 0 | 41.7% | 0.0% | 0.56 | 0.00 | 4.6 | 70.6 | 14.2 |
| 25% | 1 | 24 | 13 | 0 | 54.2% | 0.0% | 0.57 | 0.00 | 4.6 | 68.9 | 14.5 |
| 30% | 1 | 24 | 3 | 0 | 12.5% | 0.0% | 0.58 | 0.00 | 4.3 | 69.2 | 14.4 |
