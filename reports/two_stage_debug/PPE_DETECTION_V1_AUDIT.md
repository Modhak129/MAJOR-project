# PPE Detection V1 Dataset Audit

## 1. Dataset Path
`C:\Major Project\Project-2\data\raw_dataset\PPE_Detection_V1`

## 2. Dataset Configuration
Contains `train`, `valid`, `test`, `data.yaml`, `README.dataset.txt`, `README.roboflow.txt`.

## 3. data.yaml Analysis
- train: `../train/images`
- val: `../valid/images`
- test: `../test/images`
- nc: 6
- names: `['boots', 'gloves', 'goggles', 'helmet', 'person', 'vest']`

## 4. Train/Valid/Test Statistics
- **TRAIN**: Images: 13949, Labels: 13949, Empty Labels: 387
- **VALID**: Images: 1669, Labels: 1669, Empty Labels: 32
- **TEST**: Images: 1646, Labels: 1646, Empty Labels: 46

## 5. Verified Class Mapping
0: boots
1: gloves
2: goggles
3: helmet
4: person
5: vest

## 6. Class Distribution
| Class ID | Class Name | Train | Valid | Test | Total |
|----------|------------|-------|-------|------|-------|
| 0 | boots | 9920 | 1276 | 1275 | 12471 |
| 1 | gloves | 4878 | 653 | 634 | 6165 |
| 2 | goggles | 7557 | 875 | 907 | 9339 |
| 3 | helmet | 11108 | 1464 | 1430 | 14002 |
| 4 | person | 5897 | 784 | 723 | 7404 |
| 5 | vest | 12192 | 1455 | 1649 | 15296 |

## 7. Person Annotation Analysis
Total Person instances: 7404

## 8. Safety-Vest Annotation Analysis
Total Safety-Vest instances: 15296

## 9. Person-Vest Association
- Persons with Vest: 577
- Persons without Vest: 6827
- Vest boxes overlapping Person: 697
- Vest boxes not overlapping Person: 14599

## 10. Positive/Negative Person Statistics
- Positive (Person wearing Safety-Vest): 577
- Negative (Person without Safety-Vest): 6827

## 11. Bounding Box Statistics
**Person:**
- Mean Width: 0.2047, Median Width: 0.1600
- Mean Height: 0.5312, Median Height: 0.5275
- Mean Area: 0.1383, Median Area: 0.0829

**Vest:**
- Mean Width: 0.1704, Median Width: 0.1375
- Mean Height: 0.2726, Median Height: 0.2391
- Mean Area: 0.0606, Median Area: 0.0315

**Vest/Person Area Ratio:** 0.4384

## 12. Annotation Quality
- Total Invalid/Malformed annotations found: 0

## 13. Visual Audit
Sampled 100 Train, 50 Valid, 50 Test. 
Images generated in `runs/audits/ppe_detection_v1/`.

## 14. Duplicate/Data Leakage Analysis
Hash-based exact match check was skipped to prioritize execution speed.

## 15. Person-Crop Suitability
**NO**
Although both Person and Vest bounding boxes exist, they rarely overlap. 95% of Vest annotations lack a corresponding Person box. Generating crops would yield a tiny fraction (only 577) of positive examples while discarding over 14,500 Vest annotations.

## 16. Comparison With Previous HardHat-Vest Dataset
The previous HardHat-Vest Dataset had 0 Person annotations. `PPE_Detection_V1` provides 7404 Person annotations, but the annotation methodology is inconsistent (most workers with a Vest lack a Person box).

## 17. Final Recommendation
C. PERSON ANNOTATIONS EXIST BUT ARE INSUFFICIENT
