import os
import time
import shutil
import pandas as pd
import json
from ultralytics import YOLO

def main():
    expansions = [0, 10, 15, 20, 25]
    base_data_dir = r"C:\Major Project\Project-2\data\processed"
    project_dir = r"C:\Major Project\Project-2\runs\detect\stage2_vest_expansion"
    
    os.makedirs(project_dir, exist_ok=True)
    
    results_summary = {}

    for exp in expansions:
        print(f"============================================================")
        print(f"Starting training for {exp}% expansion")
        print(f"============================================================")
        
        data_yaml = os.path.join(base_data_dir, f"stage2_vest_crops_{exp}", "data.yaml")
        name = f"exp_{exp}"
        
        run_dir = os.path.join(project_dir, name)
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)
            
        model = YOLO("yolov8n.pt")
        
        start_time = time.time()
        
        results = model.train(
            data=data_yaml,
            epochs=10,
            imgsz=640,
            batch=32,
            device=0,
            patience=3,
            amp=True,
            project=project_dir,
            name=name
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        csv_path = os.path.join(run_dir, "results.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            
            fitness = df['metrics/mAP50(B)'] * 0.1 + df['metrics/mAP50-95(B)'] * 0.9
            best_idx = fitness.idxmax()
            best_row = df.loc[best_idx]
            
            results_summary[exp] = {
                "best_epoch": int(best_row['epoch']),
                "precision": float(best_row['metrics/precision(B)']),
                "recall": float(best_row['metrics/recall(B)']),
                "map50": float(best_row['metrics/mAP50(B)']),
                "map50_95": float(best_row['metrics/mAP50-95(B)']),
                "training_time": training_time
            }
        else:
            print(f"Warning: results.csv not found for {exp}% expansion")
            
    with open(os.path.join(project_dir, "results_summary.json"), "w") as f:
        json.dump(results_summary, f, indent=4)
        
    print("All micro-training experiments completed successfully.")

if __name__ == "__main__":
    main()
