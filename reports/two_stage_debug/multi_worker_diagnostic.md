# MULTI-WORKER + VEST DIAGNOSTIC

## Worker 1
Person: Detected
confidence: ~0.89 (e.g., 0.896 on frame 480)
Track ID: Successfully initialized (ID: 1)
Stage-2: Executed
Helmet confidence: ~0.76 (e.g. 0.81)
Vest confidence: N/A

## Worker 2
Person: Detected (Raw YOLO before threshold)
confidence: ~0.22 - 0.27 (e.g., 0.276 on frame 480)
Track ID: NONE
Stage-2: Skipped
Helmet confidence: Skipped
Vest confidence: Skipped

## Missing Worker Root Cause
Choose exactly one:

[x] Person detector confidence
[ ] NMS
[ ] Person filtering
[ ] ByteTrack
[ ] Worker loop only processes first worker
[ ] PPE association
[ ] ROI/geofence
[ ] Other

(Explanation: The right worker is fully detected by the raw YOLO model, but because their raw confidence hovers around 0.22–0.27, they are stripped out by the global `min_confidence = 0.45` threshold before they ever reach ByteTrack. This proves that the multi-worker loop and tracking code are perfectly fine, but the SH17 model struggles to assign high confidence to this specific person.)

## Vest Root Cause
Choose exactly one:

[ ] Vest confidence threshold
[ ] No raw vest detection
[ ] Wrong class mapping
[ ] PPE crop problem
[x] Model generalization/domain shift
[ ] PPE association
[ ] Other

(Explanation: A sweep on `indianworkers.mp4` reveals that the raw YOLO Vest confidence frequently plummets to ~0.10 or 0.11 on several frames where the vest is visibly present. Lowering the threshold to 0.10 would invite massive false positives on unrelated objects. The vest is technically "detected" by the model's innermost layers, but the model has poor generalization and assigns it a fundamentally unviable confidence score.)

## Stage-2
Executed for Worker 1: YES
Executed for Worker 2: NO (Because Worker 2 never became a tracked person)

## Important
Both issues are confirmed to be Stage-1 (SH17) model generalization issues, specifically its failure to assign robust confidence scores to the Person and Safety-vest classes in these specific video domains. The pipeline architecture itself (Stage-2, ByteTrack, Multi-worker processing, association) is working exactly as intended.
