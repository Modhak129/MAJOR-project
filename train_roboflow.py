import os
import torch
from ultralytics import YOLO

def verify_gpu():
    if not torch.cuda.is_available():
        print("CRITICAL ERROR: CUDA NOT AVAILABLE.")
        exit(1)
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

def speed_test():
    print("=== STARTING SPEED TEST ===")
    model = YOLO("yolov8n.pt") # Standard pretrained weights
    model.train(
        data="roboflow_local.yaml",
        epochs=2,
        imgsz=640,
        batch=32,
        workers=4,
        device=0,
        project="runs/detect/edge/weights/training/roboflow",
        name="speedtest_yolov8n",
        amp=True
    )
    print("=== SPEED TEST SUCCESSFUL ===")

def full_train():
    print("=== STARTING FULL TRAINING ===")
    # It will initialize from the pretrained model
    model = YOLO("yolov8n.pt")
    model.train(
        data="roboflow_local.yaml",
        epochs=30,
        imgsz=640,
        batch=32,
        workers=6,
        device=0,
        project="runs/detect/edge/weights/training/roboflow",
        name="roboflow_yolov8n_640",
        amp=True,
        save=True,
        save_period=1,
        resume=False
    )
    print("=== FULL TRAINING SUCCESSFUL ===")

if __name__ == "__main__":
    verify_gpu()
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        full_train()
    else:
        speed_test()
