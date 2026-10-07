import sys
import glob
import os
from ultralytics import YOLO

def run_val(model_path, data_yaml, imgsz=640):
    print(f"\nEvaluating {model_path} at imgsz={imgsz} on {data_yaml}")
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, imgsz=imgsz, split="val")
    
    print("\n--- PER-CLASS METRICS ---")
    class_indices = metrics.ap_class_index
    for i, c in enumerate(class_indices):
        name = model.names[c]
        p = metrics.class_result(i)[0]
        r = metrics.class_result(i)[1]
        map50 = metrics.class_result(i)[2]
        if name in ["Person", "Safety-vest", "Helmet", "Gloves"]:
            print(f"{name:<15}: P={p:.4f} R={r:.4f} mAP50={map50:.4f}")

if __name__ == "__main__":
    yaml_path = r"C:\Major Project\Project-2\sh17_local.yaml"
    search_path = r"C:\Major Project\Project-2\runs\detect\edge\weights\training\finetuned\sh17_yolov8n_640*\weights\best.pt"
    models = glob.glob(search_path)
    if not models:
        print("Model not found.")
        sys.exit(1)
    
    new_model = max(models, key=os.path.getctime)
    run_val(new_model, yaml_path, 640)
