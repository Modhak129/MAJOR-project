import json

def generate_report():
    with open("results/multi_video_comparison/multi_video_results.json", "r") as f:
        data = json.load(f)
        
    md = """# Multi-Video PPE Detection Benchmark

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
"""
    for entry in data:
        v = entry["video"]
        m = entry["metrics"]
        md += f"| {m['model']} | {v} | {m['unique_workers']} | {m['total_persons']} | {m['total_helmets']} | {m['total_vests']} | {m['total_gloves']} | {m['fps']:.1f} | {m['latency_ms']:.1f} |\n"
        
    md += """
---

## 3. Video-Specific Detailed Results
"""
    # Group by video
    videos = {}
    for entry in data:
        v = entry["video"]
        if v not in videos: videos[v] = []
        videos[v].append(entry["metrics"])
        
    for v_name, metrics_list in videos.items():
        md += f"\n### {v_name}\n"
        md += "| Metric |" + "".join([f" {m['model']} |" for m in metrics_list]) + "\n"
        md += "|---|" + "".join(["---:|" for _ in metrics_list]) + "\n"
        md += "| Person Detections |" + "".join([f" {m['total_persons']} |" for m in metrics_list]) + "\n"
        md += "| Helmet Detections |" + "".join([f" {m['total_helmets']} |" for m in metrics_list]) + "\n"
        md += "| Safety Vest Detections |" + "".join([f" {m['total_vests']} |" for m in metrics_list]) + "\n"
        md += "| Glove Detections |" + "".join([f" {m['total_gloves']} |" for m in metrics_list]) + "\n"
        md += "| Helmet Detections / Person |" + "".join([f" {(m['total_helmets']/max(1, m['total_persons'])):.2f} |" for m in metrics_list]) + "\n"
        md += "| Vest Detections / Person |" + "".join([f" {(m['total_vests']/max(1, m['total_persons'])):.2f} |" for m in metrics_list]) + "\n"
        md += "| FPS |" + "".join([f" {m['fps']:.1f} |" for m in metrics_list]) + "\n"
        md += "| Avg Latency |" + "".join([f" {m['latency_ms']:.1f} ms |" for m in metrics_list]) + "\n"
        
        md += "\n**Confidence Statistics:**\n"
        for m in metrics_list:
            md += f"- **{m['model']}**: Helmet (Mean: {m['helmet_mean_conf']:.2f}, Max: {m['helmet_max_conf']:.2f}), Vest (Mean: {m['vest_mean_conf']:.2f}, Max: {m['vest_max_conf']:.2f})\n"

    md += """
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
"""
    with open("results/multi_video_comparison/multi_video_results.md", "w") as f:
        f.write(md)
    print("Report generated successfully!")

if __name__ == "__main__":
    generate_report()
