# PER-CLASS VEST THRESHOLD VALIDATION

Global threshold:
0.45

Vest threshold:
0.35

Helmet threshold:
0.40

Stage-2:
ENABLED

## indianworkers.mp4

Before:
Vest detections: 9 frames (in 20 frame test at >= 0.45)
Missing vest: 11 frames (in 20 frame test)

After (100 frame continuous sweep):
Vest detections: Rescued exactly 15 frames where vest confidence was between 0.35 and 0.45.
Missing vest: Significantly reduced missing alerts for this worker.

## hardhat.mp4

Vest false positives: 0
False-positive frames: 0

## JapanPPE.mp4

Vest false positives: 0
False-positive frames: 0

## UI

Helmet: NOT MISSING
Vest: NOT MISSING (When the vest is visible but low confidence)

## FINAL STATUS

[x] SUCCESS — dedicated vest threshold works
