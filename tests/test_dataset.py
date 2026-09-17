import os
from pathlib import Path
from ultralytics import YOLO

def test_dataset_partitions():
    print("=== Testing Real Dataset Partitions ===")
    
    clients = [1, 2]
    total_images = 0
    total_labels = 0
    
    for client_id in clients:
        client_dir = Path(f"data/partitioned/client_{client_id}")
        
        # Check directories exist
        assert client_dir.exists(), f"Client {client_id} directory missing."
        
        # Check yaml exists
        yaml_path = client_dir / "dataset.yaml"
        assert yaml_path.exists(), f"dataset.yaml missing for Client {client_id}."
        
        # Count images and labels
        images = list((client_dir / "images" / "train").glob("*.jpg"))
        labels = list((client_dir / "labels" / "train").glob("*.txt"))
        
        num_images = len(images)
        num_labels = len(labels)
        
        print(f"Client {client_id}: {num_images} images, {num_labels} labels found.")
        
        assert num_images > 1000, f"Expected >1000 images for client {client_id}, found {num_images}."
        
        total_images += num_images
        total_labels += num_labels

    print(f"Total partitioned images: {total_images}")
    assert total_images > 2500, "Dataset partitioning missed images."
    print("✅ Partition verification passed.\n")

def test_yolo_training_init():
    print("=== Testing YOLOv8 Training Initialization ===")
    
    # We load a nano model and train for 1 epoch on a few batches to ensure
    # the dataset.yaml is perfectly valid and readable by the Ultralytics engine.
    model = YOLO("yolov8n.pt")
    
    # The dataset.yaml inside data/partitioned has absolute paths for Docker (/app/data/...)
    # But since we are testing on the host, we need a temporary host-compatible yaml
    with open("data/partitioned/client_1/dataset.yaml", "r") as f:
        content = f.read()
    
    # Replace the docker absolute path with the local absolute path for this test
    host_yaml = content.replace("/app/data", str(Path("data").absolute()).replace('\\', '/'))
    
    with open("data/test_dataset.yaml", "w") as f:
        f.write(host_yaml)
        
    try:
        # Run a micro-training session (1 epoch, minimal workers)
        print("Starting micro-training session...")
        model.train(
            data="data/test_dataset.yaml",
            epochs=1,
            imgsz=320,
            batch=2,
            workers=0, # Disable multiprocessing for quick test
            project="data/test_runs",
            name="test_train"
        )
        print("✅ YOLOv8 successfully initialized and trained on the real dataset!")
    except Exception as e:
        print(f"❌ YOLOv8 training failed: {e}")
        raise e
    finally:
        # Cleanup
        if os.path.exists("data/test_dataset.yaml"):
            os.remove("data/test_dataset.yaml")

if __name__ == "__main__":
    test_dataset_partitions()
    test_yolo_training_init()
