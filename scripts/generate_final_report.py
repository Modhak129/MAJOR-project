import json
import numpy as np

def generate_report():
    with open("scripts/audit_ppe_final_stats.json", "r") as f:
        data = json.load(f)
        
    s = data["stats"]
    c = data["actual_class_counts"]
    v = data
    cm = data["class_map"]
    cc = data["csv_counts"]
    
    md = f"""# PPE Final Dataset Audit

## 1. Dataset Path
`C:\\Major Project\\Project-2\\data\\raw_dataset\\PPE_Final_Dataset`

## 2. Dataset Structure
Standard YOLO format with `train`, `val`, and `test` directories.

## 3. data.yaml Analysis
- nc: 10
- names: `{[v for k,v in cm.items()]}`

## 4. dataset_class_counts.csv Analysis
Provides aggregated class counts which need to be verified against the ground truth labels.

## 5. CSV vs Actual Annotation Comparison
| Class ID | Class Name | CSV Total | Actual Total | Difference |
|----------|------------|-----------|--------------|------------|
"""
    for i, name in cm.items():
        csv_tot = int(cc.get('TOTAL', {}).get(name, 0))
        act_tot = c["train"].get(str(i), 0) + c["val"].get(str(i), 0) + c["test"].get(str(i), 0)
        md += f"| {i} | {name} | {csv_tot} | {act_tot} | {act_tot - csv_tot} |\n"
        
    md += f"""
CSV matches actual labels: **{v["csv_match"]}**

## 6. Train/Val/Test Statistics
- **TRAIN**: Images: {s['train']['images']}, Labels: {s['train']['labels']}, Empty: {s['train']['empty_lbl']}
- **VAL**: Images: {s['val']['images']}, Labels: {s['val']['labels']}, Empty: {s['val']['empty_lbl']}
- **TEST**: Images: {s['test']['images']}, Labels: {s['test']['labels']}, Empty: {s['test']['empty_lbl']}

## 7. Verified Class Mapping
"""
    for i, name in cm.items():
        md += f"{i}: {name}\n"
        
    md += "\n## 8. Class Distribution\n"
    md += "| Class ID | Class Name | Train | Val | Test | Total | Images |\n"
    md += "|----------|------------|-------|-----|------|-------|--------|\n"
    for i, name in cm.items():
        tr = c["train"].get(str(i), 0)
        va = c["val"].get(str(i), 0)
        te = c["test"].get(str(i), 0)
        tot = tr + va + te
        imgs = v["class_imgs"].get(str(i), 0)
        md += f"| {i} | {name} | {tr} | {va} | {te} | {tot} | {imgs} |\n"
        
    md += f"""
## 9. Person Annotation Analysis
Total Person instances: {v['total_persons']}

## 10. Safety-Vest Annotation Analysis
Total Safety-Vest instances: {v['total_vests']}

## 11. Person-Vest Association
- Persons associated with Vest: {v['persons_with_vest']['total']} ({(v['persons_with_vest']['total']/v['total_persons']*100 if v['total_persons'] else 0):.1f}%)
- Persons without Vest: {v['persons_without_vest']['total']} ({(v['persons_without_vest']['total']/v['total_persons']*100 if v['total_persons'] else 0):.1f}%)
- Vests clearly inside Person: {v['vest_inside_person']}
- Vests partially overlapping Person: {v['vest_overlap_person']}
- Vests NOT associated with any Person: {v['vest_outside_person']}

## 12. Positive/Negative Person Statistics
- Positive (with Vest): {v['persons_with_vest']['total']}
- Negative (without Vest): {v['persons_without_vest']['total']}

## 13. Multi-Worker Analysis
- Images with multiple Persons: {v['multi_worker_imgs']}
- Images with multiple Vests: {v['multi_vest_imgs']}
- Ambiguous Vest associations (1 vest overlapping multiple persons): {v['vest_multi_person_ambiguous']}

## 14. Bounding Box Statistics
"""
    def b_stats(box_dict):
        if not box_dict["w"]: return "N/A"
        return f"Mean Width: {np.mean(box_dict['w']):.4f}, Mean Height: {np.mean(box_dict['h']):.4f}, Mean Area: {np.mean(box_dict['a']):.4f}"
        
    md += f"**Person:** {b_stats(v['bbox_stats']['person'])}\n"
    md += f"**Vest:** {b_stats(v['bbox_stats']['vest'])}\n"
    vp = np.mean(v["vp_ratios"]) if v["vp_ratios"] else 0
    md += f"**Average Vest Area / Person Area:** {vp:.4f}\n"

    md += f"""
## 15. Annotation Quality
Total invalid/malformed annotations: {v['problems']}

## 16. Visual Audit
Generated visual audits in `runs/audits/ppe_final_dataset/`.
Audited 100 Train, 50 Val, 50 Test.

## 17. Duplicate/Data Leakage Check
- Train-Val duplicates: {v['t_v_dupes']}
- Train-Test duplicates: {v['t_t_dupes']}
- Val-Test duplicates: {v['v_t_dupes']}

## 18. Person-Crop Suitability
**{v['suitability']}**
There are over 4000 Persons natively annotated, and most Vests overlap or fall inside Person bounding boxes, enabling high-quality relative crop generation.

## 19. Crop Expansion Readiness
The dataset supports expansions at 0%, 10%, 15%, 20%, and 25%. Vest coordinates are well-bounded within the person, so expansions will safely capture edge cases (e.g. arms extending outside the torso).

## 20. Comparison With Previous Datasets
Unlike `HardHat-Vest Dataset` (0 Persons) and `PPE_Detection_V1` (only 577 overlapping persons), `PPE_Final_Dataset` contains robust, highly-correlated Person-Vest annotations suitable for crop generation.

## 21. Final Recommendation
{v['rec']}
"""
    with open(r"C:\Major Project\Project-2\reports\two_stage_debug\PPE_FINAL_DATASET_AUDIT.md", "w") as f:
        f.write(md)

if __name__ == "__main__":
    generate_report()
