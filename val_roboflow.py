import os
from ultralytics import YOLO

def main():
    print("Evaluating Roboflow Model on Roboflow Validation Set...")
    model = YOLO("runs/detect/edge/weights/training/roboflow/roboflow_yolov8n_640/weights/best.pt")
    res = model.val(data="roboflow_local.yaml", split="val", imgsz=640, device=0)
    print("Roboflow Validation complete.")
    
if __name__ == "__main__":
    main()
