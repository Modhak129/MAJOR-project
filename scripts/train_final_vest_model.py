import os
from ultralytics import YOLO

def main():
    data_yaml = r"C:\Major Project\Project-2\data\processed\stage2_vest_crops_20\data.yaml"
    project_dir = r"C:\Major Project\Project-2\runs\detect"
    name = "stage2_vest_final_20pct"
    
    # 1. Start from yolov8n.pt
    model = YOLO("yolov8n.pt")
    
    # 2. Train for 50 epochs
    results = model.train(
        data=data_yaml,
        epochs=50,
        imgsz=640,
        batch=32,
        device=0,
        workers=2,
        amp=True,
        patience=10, # Reasonable patience for 50 epochs
        project=project_dir,
        name=name
    )
    
    print("Final training completed successfully.")

if __name__ == "__main__":
    main()
