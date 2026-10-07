import time
import torch
import cv2
import numpy as np
from ultralytics import YOLO

# Export models to ONNX
vest_model = YOLO("edge/weights/stage2_vest_yolov8n_640_best.pt")
helmet_model = YOLO("edge/weights/hardhat_helmet_yolov8n_640_best.pt")

print("Exporting to ONNX...")
vest_onnx_path = vest_model.export(format="onnx", half=True, dynamic=True)
helmet_onnx_path = helmet_model.export(format="onnx", half=True, dynamic=True)

# Load ONNX models
vest_onnx = YOLO(vest_onnx_path)
helmet_onnx = YOLO(helmet_onnx_path)

# Create dummy crops (e.g. 5 crops of size 200x150)
crops = [np.random.randint(0, 255, (200, 150, 3), dtype=np.uint8) for _ in range(5)]

print("Testing ONNX batch inference vs PyTorch batch inference...")

# Warmup
for _ in range(3):
    vest_model.predict(crops, verbose=False, device='0')
    vest_onnx.predict(crops, verbose=False, device='0')

# Benchmark PyTorch
t0 = time.time()
for _ in range(100):
    vest_model.predict(crops, verbose=False, device='0')
pt_time = (time.time() - t0) / 100

# Benchmark ONNX
t0 = time.time()
for _ in range(100):
    vest_onnx.predict(crops, verbose=False, device='0')
onnx_time = (time.time() - t0) / 100

print(f"PyTorch batch of 5: {pt_time*1000:.2f} ms")
print(f"ONNX batch of 5: {onnx_time*1000:.2f} ms")
