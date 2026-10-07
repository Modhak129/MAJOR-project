from ultralytics import YOLO
import torch
import os

def main():
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Tiny tensor test
        x = torch.rand(3, 3).cuda()
        print("CUDA Tensor Test:", x.shape)
    else:
        print("CUDA IS NOT AVAILABLE. STOPPING.")
        return

    # Load YOLOv8n SH17 best model
    model_path = r"edge\weights\sh17_yolov8s_best.pt"
    model = YOLO(model_path)
    
    data_yaml = r"C:\Major Project\Project-2\sh17_local.yaml"
    save_dir = r"edge\weights\training\finetuned"
    
    # Create save dir if not exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Train
    print("Starting YOLO Fine-Tuning...")
    model.train(
        data=data_yaml,
        epochs=30, # 1 epoch for fast dry-run / pipeline verification
        imgsz=640,
        device=0,
        batch=48, # Batch 32 should comfortably fit in 16GB VRAM for YOLOv8n
        workers=6, # Safe for Windows multiprocessing

        # cache="disk",
        
        project=save_dir,
        name="sh17_yolov8s_640",
        exist_ok=True,
        # Augmentation Settings (avoid extreme)
        hsv_h=0.015,
        hsv_s=0.5, # Reduced color augmentation slightly to rely on shape, but keep some for robustness
        hsv_v=0.4,
        degrees=5.0, # Slight rotation
        translate=0.1,
        scale=0.3,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0, # Mosaic helps with small objects
        mixup=0.0,
        copy_paste=0.0,
        amp=True ,# Mixed precision
        save=True,
        save_period=5,
        plots=True
    )

if __name__ == "__main__":
    main()
