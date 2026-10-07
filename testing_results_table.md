# Real-Time Edge-AI Safety Compliance & Monitoring System
## Testing Results (Pytest Implementation)

| Test Case ID | Name | Expected Result | Actual Result | Pass/Fail |
| :--- | :--- | :--- | :--- | :--- |
| **FLM-02** | FedAvg Aggregation (`test_fedavg.py`) | Output matches expected FedAvg weighted average given known client weight updates. | Parameter updates correctly aggregated to weighted average `[2.0, 2.0]` for equal-weighted clients. | Pass |
| **ARM-02** | Adaptive Model Assignment (`test_adaptive.py`) | Correct model variant (Nano or Small) is assigned based on VRAM threshold (4096 MB). | YOLOv8-Nano assigned for 2048 MB VRAM (< 4GB); YOLOv8-Small assigned for 6144 MB (≥ 4GB). | Pass |
| **ARM-05** | Gradient Sparsification (`test_adaptive.py`) | Top-k sparsification (k=0.1) retains top 10% magnitude weights and zeroes out 90% of tensor values. | Exactly 10 out of 100 values retained (90% zeroed out) meeting top-k magnitude threshold. | Pass |
| **ANM-02** | Alert Schema Validation (`test_alerts.py`) | Alert payload generated via POST strictly contains `client_id`, `type`, `confidence`, and `timestamp`. | Valid JSON payload generated and POSTed with all required schema fields present. | Pass |
| **ANM-03** | Alert Cooldown Suppression (`test_alerts.py`) | Duplicate alerts for identical violation type are suppressed within cooldown period (2s). | Duplicate alert suppressed during cooldown; successfully dispatched after 2.1s cooldown or upon alert type change. | Pass |
| **EIM-01** | Inference Engine Initialization (`test_inference.py`) | InferenceEngine properly instantiates the underlying YOLO model with specified weights path. | YOLO model initialized successfully with configured weight file (`mock_path.pt`). | Pass |
| **EIM-06** | Confidence Filtering & Alert Trigger (`test_inference.py`) | Detections ≥ 0.45 trigger alerts; detections below the 0.45 minimum threshold are rejected. | Alert sent for 0.85 confidence PPE violation; alert successfully suppressed for 0.20 low-confidence detection. | Pass |
| **FGI-01** | Missing PPE Detection Logic (`test_fine_grained_inference.py`) | Person bounding box lacking mandatory safety gear (gloves) generates a missing PPE alert. | Alert triggered for "Missing PPE" with `gloves` correctly flagged in `missing_items`. | Pass |
| **FGI-02** | Geofence Boundary Intrusion (`test_fine_grained_inference.py`) | Limb bounding box (hand/leg) intersecting restricted polygon zone triggers intrusion alert. | Zone Intrusion alert dispatched identifying "Hand" breach into restricted coordinate polygon. | Pass |
| **DAT-01** | Dataset Partition Verification (`test_dataset.py`) | Partitioned client folders contain valid `dataset.yaml` and meet minimum image/label thresholds. | Client 1 & 2 partitions verified with >1000 images per client and >2500 total images. | Pass |
| **DAT-02** | YOLO Training Initialization (`test_dataset.py`) | Ultralytics YOLOv8 engine initializes micro-training session on partitioned dataset config. | Model initializes and completes 1 epoch training run without data pipeline errors. | Pass |

