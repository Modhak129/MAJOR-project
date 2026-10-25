import os
import yaml
import shutil
import random
from pathlib import Path

def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def partition_real_dataset():
    config = load_config()
    random.seed(config["data"]["random_seed"])
    num_clients = config["fl"]["num_clients"]
    
    raw_dataset_dir = Path("data/raw_dataset/css-data")
    train_images_dir = raw_dataset_dir / "train" / "images"
    train_labels_dir = raw_dataset_dir / "train" / "labels"
    
    if not train_images_dir.exists():
        print(f"Error: Dataset not found at {train_images_dir}.")
        return

    # Get all training images
    all_images = list(train_images_dir.glob("*.jpg"))
    random.shuffle(all_images)
    
    print(f"Found {len(all_images)} training images. Partitioning across {num_clients} clients...")
    
    # Clean output directory
    partitioned_dir = Path("data/partitioned")
    if partitioned_dir.exists() and (partitioned_dir / "client_1" / "dataset.yaml").exists():
        print("Dataset already partitioned. Skipping to prevent race conditions.")
        return
        
    chunk_size = len(all_images) // num_clients
    
    classes = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']
    
    for client_id in range(1, num_clients + 1):
        client_dir = partitioned_dir / f"client_{client_id}"
        client_train_images = client_dir / "images" / "train"
        client_train_labels = client_dir / "labels" / "train"
        
        client_train_images.mkdir(parents=True, exist_ok=True)
        client_train_labels.mkdir(parents=True, exist_ok=True)
        
        start_idx = (client_id - 1) * chunk_size
        end_idx = start_idx + chunk_size if client_id < num_clients else len(all_images)
        client_subset = all_images[start_idx:end_idx]
        
        print(f"Client {client_id}: Assigning {len(client_subset)} images.")
        
        for img_path in client_subset:
            # Copy image
            shutil.copy(img_path, client_train_images / img_path.name)
            
            # Copy corresponding label
            label_name = img_path.stem + ".txt"
            label_path = train_labels_dir / label_name
            if label_path.exists():
                shutil.copy(label_path, client_train_labels / label_name)
                
        # Generate ultralytics dataset YAML for the client
        # Paths are configured for the Docker container filesystem (/app/data/...)
        yaml_content = {
            "path": f"/app/data",
            "train": f"partitioned/client_{client_id}/images/train",
            "val": "raw_dataset/css-data/valid/images",  # Shared validation set
            "names": {i: name for i, name in enumerate(classes)}
        }
        
        with open(client_dir / "dataset.yaml", "w") as f:
            yaml.dump(yaml_content, f, sort_keys=False)
            
    print("Real dataset partitioned successfully.")

if __name__ == "__main__":
    partition_real_dataset()
