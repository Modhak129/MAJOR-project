import os
from ultralytics import YOLO

def main():
    print("Evaluating SH17 Model on SH17 Validation Set...")
    model = YOLO("edge/weights/sh17_yolov8n_best.pt")
    # Assuming sh17_local.yaml exists from previous experiments
    res = model.val(data="sh17_local.yaml", split="val", imgsz=640, device=0)
    print("SH17 Validation complete.")
    
if __name__ == "__main__":
    main()
