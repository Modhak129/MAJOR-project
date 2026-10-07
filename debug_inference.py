import cv2
from ultralytics import YOLO

def debug_inference():
    model_path = "edge/weights/sh17_yolov8n_best.pt"
    image_path = "C:/Users/Vishnu/.gemini/antigravity/brain/e356b07e-349e-4459-aa61-1ee9b611ff41/.user_uploaded/media_1790161280279.png"
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    print("\n--- MODEL CLASS MAPPING ---")
    print(model.names)
    print("---------------------------\n")
    
    print(f"Running inference on: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print("Failed to load image!")
        return

    # Run inference with a low threshold to see EVERYTHING
    results = model(frame, conf=0.1, verbose=False)[0]
    
    print("--- RAW MODEL OUTPUT ---")
    detected_items = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = model.names[cls_id]
        x1, y1, x2, y2 = [float(c) for c in box.xyxy[0]]
        
        detected_items.append({
            "class_id": cls_id,
            "class_name": cls_name,
            "conf": conf,
            "box": (x1, y1, x2, y2)
        })
        
        print(f"{cls_name:<15} {conf:.2f}  bbox: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
        
        # Draw on frame
        color = (0, 255, 0) if cls_name == "Person" else (255, 0, 0)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"{cls_name} {conf:.2f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imwrite("debug_annotated.png", frame)
    print("\nAnnotated frame saved to debug_annotated.png")

if __name__ == "__main__":
    debug_inference()
