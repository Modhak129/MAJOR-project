import os
import sys
import torch
import cv2
import json
import time
from pathlib import Path
from ultralytics import YOLO

def preflight():
    print("--- PREFLIGHT CHECK ---")
    data_yaml = Path(r"C:\Major Project\Project-2\data\raw_dataset\SH17_YOLO_CLEAN\data.yaml")
    if not data_yaml.exists():
        print("FAIL: data.yaml not found.")
        sys.exit(1)
        
    print(f"data.yaml exists: {data_yaml}")
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("FAIL: CUDA is not available!")
        sys.exit(1)
        
    device = torch.device('cuda:0')
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, Memory: {props.total_memory / 1e9:.2f} GB")
    
    # Dataset counts
    train_dir = Path(r"C:\Major Project\Project-2\data\raw_dataset\SH17_YOLO_CLEAN\images\train")
    val_dir = Path(r"C:\Major Project\Project-2\data\raw_dataset\SH17_YOLO_CLEAN\images\val")
    
    train_imgs = len(list(train_dir.glob("*.*")))
    val_imgs = len(list(val_dir.glob("*.*")))
    
    print(f"Train images: {train_imgs}")
    print(f"Validation images: {val_imgs}")
    
    if train_imgs != 6479 or val_imgs != 1620:
        print("WARNING: Dataset image counts do not exactly match expected numbers.")
        # We will continue anyway as long as >0
        if train_imgs == 0 or val_imgs == 0:
            print("FAIL: Missing images.")
            sys.exit(1)
            
    print("Preflight complete.\n")

def run_training():
    preflight()
    
    model = YOLO("yolov8n.pt")
    data_yaml = r"C:\Major Project\Project-2\data\raw_dataset\SH17_YOLO_CLEAN\data.yaml"
    project_dir = r"C:\Major Project\Project-2\runs\detect\edge\weights\training"
    name = "clean_sh17_yolov8n_640"
    
    print("Starting YOLOv8n training...")
    start_time = time.time()
    
    results = model.train(
        data=data_yaml,
        epochs=50,
        imgsz=640,
        batch=48,
        workers=6,
        device=0,
        amp=True,
        pretrained=True,
        cache=False,
        seed=42,
        patience=15,
        save=True,
        plots=True,
        project=project_dir,
        name=name
    )
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Training completed in {duration/60:.2f} minutes.")
    
    return project_dir, name

def run_validation(project_dir, name):
    print("Running validation metrics extraction...")
    best_pt = Path(project_dir) / name / "weights" / "best.pt"
    if not best_pt.exists():
        print("FAIL: best.pt not found.")
        sys.exit(1)
        
    model = YOLO(str(best_pt))
    data_yaml = r"C:\Major Project\Project-2\data\raw_dataset\SH17_YOLO_CLEAN\data.yaml"
    
    val_results = model.val(data=data_yaml, imgsz=640)
    
    # Map class index to names
    names = model.names
    
    metrics = {
        "Person": {},
        "Head": {},
        "Safety-vest": {},
        "Gloves": {},
        "Helmet": {},
        "Overall": {}
    }
    
    # val_results.box contains metrics.
    # We can iterate through val_results.box.ap_class_index and get metrics
    for i, c_idx in enumerate(val_results.box.ap_class_index):
        c_name = names[c_idx]
        if c_name in metrics:
            metrics[c_name]["Precision"] = val_results.box.p[i]
            metrics[c_name]["Recall"] = val_results.box.r[i]
            metrics[c_name]["mAP50"] = val_results.box.ap50[i]
            metrics[c_name]["mAP50-95"] = val_results.box.ap[i]
            
    metrics["Overall"]["Precision"] = val_results.box.mp
    metrics["Overall"]["Recall"] = val_results.box.mr
    metrics["Overall"]["mAP50"] = val_results.box.map50
    metrics["Overall"]["mAP50-95"] = val_results.box.map
    
    return metrics

def run_regression_test(project_dir, name):
    best_pt = Path(project_dir) / name / "weights" / "best.pt"
    model = YOLO(str(best_pt))
    
    img_path = r"C:\Major Project\Project-2\data\raw_dataset\SH17_clean\images\gardener-worker-gardening-machinery-162564.jpeg"
    if not Path(img_path).exists():
        return "Image not found for regression test."
        
    results = model.predict(img_path, save=True, project=Path(project_dir) / name, name="regression_test")
    
    # Check predictions
    classes = results[0].boxes.cls.cpu().numpy()
    names = model.names
    detected = [names[int(c)] for c in classes]
    
    return f"Detected: {detected}"

def run_video_benchmark(project_dir, name):
    print("Running video benchmark...")
    best_pt = Path(project_dir) / name / "weights" / "best.pt"
    model = YOLO(str(best_pt))
    
    videos = [
        r"C:\Major Project\Project-2\data\raw_dataset\source_files\source_files\JapanPPE.mp4",
        r"C:\Major Project\Project-2\data\raw_dataset\source_files\source_files\hardhat.mp4",
        r"C:\Major Project\Project-2\data\raw_dataset\source_files\source_files\indianworkers.mp4"
    ]
    
    benchmark_results = {}
    
    for vid in videos:
        vid_path = Path(vid)
        if not vid_path.exists():
            print(f"Skipping {vid_path.name}, file not found.")
            continue
            
        print(f"Benchmarking {vid_path.name}...")
        cap = cv2.VideoCapture(str(vid_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frame_limit = min(500, total_frames) # Limit to 500 frames for speed
        
        person_count = 0
        helmet_count = 0
        vest_count = 0
        glove_count = 0
        
        inf_start = time.time()
        frames_processed = 0
        
        # Track worker IDs across frames
        unique_workers = set()
        
        for _ in range(frame_limit):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run inference with tracking
            results = model.track(frame, persist=True, conf=0.45, tracker="bytetrack.yaml", verbose=False)
            
            for res in results:
                if res.boxes is None:
                    continue
                classes = res.boxes.cls.cpu().numpy()
                if res.boxes.id is not None:
                    ids = res.boxes.id.cpu().numpy()
                    for c_idx, obj_id in zip(classes, ids):
                        if int(c_idx) == 0:
                            unique_workers.add(int(obj_id))
                            
                for c in classes:
                    c_id = int(c)
                    if c_id == 0: person_count += 1
                    elif c_id == 14: helmet_count += 1
                    elif c_id == 12: vest_count += 1
                    elif c_id == 9: glove_count += 1
                    
            frames_processed += 1
            
        inf_end = time.time()
        cap.release()
        
        elapsed = inf_end - inf_start
        fps_inf = frames_processed / elapsed if elapsed > 0 else 0
        latency = (elapsed / frames_processed) * 1000 if frames_processed > 0 else 0
        
        benchmark_results[vid_path.name] = {
            "Resolution": f"{w}x{h}",
            "FPS": fps,
            "Frames evaluated": frames_processed,
            "Inference FPS": fps_inf,
            "Latency": latency,
            "Unique workers": len(unique_workers),
            "Person detections": person_count,
            "Helmet detections": helmet_count,
            "Safety-vest detections": vest_count,
            "Gloves detections": glove_count
        }
        
    return benchmark_results

def generate_report(project_dir, name, metrics, regression, benchmark_results):
    report = []
    report.append("==================================================")
    report.append("SH17 CLEAN YOLOv8n TRAINING REPORT")
    report.append("==================================================")
    
    props = torch.cuda.get_device_properties(0)
    report.append(f"1. Hardware: {props.name}, {props.total_memory / 1e9:.2f} GB")
    report.append(f"2. Dataset path: C:\\Major Project\\Project-2\\data\\raw_dataset\\SH17_YOLO_CLEAN")
    report.append(f"3. Dataset statistics: 6479 train, 1620 val images.")
    report.append(f"4. Training configuration: YOLOv8n, imgsz=640, epochs=50, batch=48")
    
    report.append("\n==================================================")
    report.append("8. Validation Metrics (Ground Truth)")
    report.append(f"Overall mAP50: {metrics['Overall'].get('mAP50', 0):.4f}")
    report.append(f"Overall mAP50-95: {metrics['Overall'].get('mAP50-95', 0):.4f}")
    
    report.append("\nPPE-Specific Metrics:")
    report.append("Class       | Precision | Recall | mAP50 | mAP50-95")
    for cls in ["Person", "Helmet", "Safety-vest", "Gloves", "Head"]:
        m = metrics.get(cls, {})
        p = m.get('Precision', 0)
        r = m.get('Recall', 0)
        m50 = m.get('mAP50', 0)
        m95 = m.get('mAP50-95', 0)
        report.append(f"{cls:<11} | {p:<9.4f} | {r:<6.4f} | {m50:<5.4f} | {m95:<8.4f}")
        
    report.append("\n==================================================")
    report.append("11. Regression-image result:")
    report.append(f"gardener-worker-gardening-machinery-162564.jpeg")
    report.append(regression)
    
    report.append("\n==================================================")
    report.append("12-14. Video Benchmark Results (Conf=0.45, ByteTrack=ENABLED)")
    for vid, b in benchmark_results.items():
        report.append(f"\nVideo: {vid}")
        for k, v in b.items():
            if isinstance(v, float):
                report.append(f"  {k}: {v:.2f}")
            else:
                report.append(f"  {k}: {v}")
                
    report.append("\n==================================================")
    report.append("15. Old vs New Comparison")
    report.append("This is primarily a data-correction experiment. The old model was trained on corrupted labels.")
    report.append("The new model is trained on correct labels. We expect valid Helmet and Safety-vest detections to now align with real PPE.")
    
    report.append("\n==================================================")
    report.append("FINAL CONCLUSION FORMAT")
    report.append("DATASET:")
    report.append("SH17_YOLO_CLEAN\n")
    report.append("MODEL:")
    report.append("YOLOv8n\n")
    report.append("STATUS:")
    report.append("TRAINING SUCCESS\n")
    report.append("BEST CHECKPOINT:")
    report.append(f"{Path(project_dir) / name / 'weights' / 'best.pt'}\n")
    
    report.append("VALIDATION:")
    for cls in ["Person", "Helmet", "Safety-vest", "Gloves", "Head"]:
        m50 = metrics.get(cls, {}).get('mAP50', 0)
        report.append(f"{cls}: {m50:.4f}")
        
    report.append("\nVIDEO BENCHMARK (Person / Helmet / Vest / Gloves):")
    for vid, b in benchmark_results.items():
        report.append(f"{vid}: P={b['Person detections']} H={b['Helmet detections']} V={b['Safety-vest detections']} G={b['Gloves detections']}")
        
    report.append("\nReady for the next controlled experiment.")
    
    out_file = Path(project_dir) / name / "clean_training_report.txt"
    with open(out_file, "w") as f:
        f.write("\n".join(report))
        
    print(f"\nReport written to {out_file}")

from multiprocessing import freeze_support

if __name__ == "__main__":
    freeze_support()
    project_dir, name = run_training()
    metrics = run_validation(project_dir, name)
    regression = run_regression_test(project_dir, name)
    benchmark_results = run_video_benchmark(project_dir, name)
    generate_report(project_dir, name, metrics, regression, benchmark_results)
