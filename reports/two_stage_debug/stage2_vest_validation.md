# STAGE-2 VEST MODEL VALIDATION

## Dataset
Images: ~22,141
Annotations: ~189,712
Vest instances: 8,131
Person instances: 0

## Dataset audit
Invalid labels: None
Class mapping:
0: helmet
1: vest
2: head
3: person

Visual audit: FAIL

## Derived person-crop dataset
Positive crops: N/A
Negative crops: N/A
Crop expansion: N/A
Train: N/A
Validation: N/A

## Model
Architecture: N/A
Image size: N/A
Batch: N/A
Epochs: N/A
Best epoch: N/A

## Validation
Precision: N/A
Recall: N/A
mAP50: N/A
mAP50-95: N/A

## Target videos

JapanPPE:
SH17 vest detection rate: N/A
Stage-2 vest detection rate: N/A

hardhat:
SH17 vest detection rate: N/A
Stage-2 vest detection rate: N/A

indianworkers:
SH17 vest detection rate: N/A
Stage-2 vest detection rate: N/A

## False positives
Stage-2 vest false positives: N/A
Best threshold: N/A

## Decision

Choose exactly one:

[ ] Stage-2 Vest is clearly better → proceed to integration
[ ] Stage-2 Vest is similar → keep SH17
[ ] Stage-2 Vest is worse → do not integrate
[x] Results are inconclusive and additional testing is required.

IMPORTANT NOTE:
The `HardHat-Vest Dataset` completely lacks Person annotations (Class 3 has 0 instances). The Stage-2 detection architecture fundamentally requires training on Person-cropped images so the model learns relative features. Without ground-truth Person bounding boxes in the raw dataset, it is impossible to generate the derived Stage-2 training dataset. 

I have stopped execution as instructed to prevent training on invalid or corrupted transformations. Pseudo-labeling was strictly forbidden by your previous instructions, so I cannot synthetically generate Person boxes using the SH17 model.

We must either acquire a dataset with bounding boxes for both Persons and Safety-Vests, or authorize pseudo-labeling.
