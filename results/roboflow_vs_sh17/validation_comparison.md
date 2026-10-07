# Validation Comparison

Metric | SH17 YOLOv8n (css-data valid) | Roboflow YOLOv8n (Roboflow valid)
---|---|---
Overall Precision | 0.005 | 0.696
Overall Recall | 0.108 | 0.796
Overall mAP50 | 0.004 | 0.760
Overall mAP50-95 | 0.001 | 0.489

### Shared Semantic Classes (mAP50)

| Class | SH17 mAP50 | Roboflow mAP50 | SH17 Recall | Roboflow Recall |
|---|---|---|---|---|
| Person | 0.001 | 0.920 | 0.139 | 0.906 |
| Helmet | 0.000 | 0.896 | 0.000 | 0.887 |
| Safety Vest | 0.000 | 0.677 | 0.000 | 0.652 |
| Gloves | 0.001 | 0.945 | 0.048 | 0.946 |
