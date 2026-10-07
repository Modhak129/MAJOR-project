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
        map50_95 = metrics.class_result(i)[3]
        if name in ["Person", "Safety-vest", "Helmet", "Gloves"]:
            print(f"{name:<15}: P={p:.4f} R={r:.4f} mAP50={map50:.4f} mAP50-95={map50_95:.4f}")

if __name__ == "__main__":
    yaml_path = r"C:\Major Project\Results\federated_industrial_safety\work\sh17_normalized.yaml"
    run_val("edge/weights/sh17_yolov8n_best.pt", yaml_path, 640)
    run_val("edge/weights/sh17_yolov8s_best.pt", yaml_path, 640)
