import os
import glob
from collections import defaultdict

def main():
    dataset_dir = r"C:\Major Project\Project-2\data\raw_dataset\HardHat-Vest Dataset"
    labels_dir = os.path.join(dataset_dir, "labels")
    images_dir = os.path.join(dataset_dir, "images")
    
    classes_file = os.path.join(labels_dir, "classes.txt")
    if not os.path.exists(classes_file):
        print(f"ERROR: {classes_file} not found.")
        return
        
    with open(classes_file, "r") as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
        
    print("============================================================")
    print("DATASET AUDIT")
    print("============================================================")
    print("Verified class mapping:")
    class_map = {}
    for i, c in enumerate(classes):
        print(f"{i} = {c}")
        class_map[i] = c
        
    print("\nAuditing splits...")
    splits = ["train", "val", "test"]
    total_images = 0
    total_labels = 0
    class_counts = defaultdict(int)
    
    for split in splits:
        split_img_dir = os.path.join(images_dir, split)
        split_lbl_dir = os.path.join(labels_dir, split)
        
        if not os.path.exists(split_img_dir) or not os.path.exists(split_lbl_dir):
            print(f"Split {split} missing.")
            continue
            
        imgs = glob.glob(os.path.join(split_img_dir, "*.jpg"))
        lbls = glob.glob(os.path.join(split_lbl_dir, "*.txt"))
        
        print(f"Split {split}: {len(imgs)} images, {len(lbls)} labels.")
        total_images += len(imgs)
        total_labels += len(lbls)
        
        # Count classes
        for lbl in lbls:
            with open(lbl, "r") as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        class_counts[cls_id] += 1
                        
    print("\nClass distribution:")
    for cls_id, count in class_counts.items():
        print(f"Class {cls_id} ({class_map.get(cls_id, 'UNKNOWN')}): {count} instances")
        
    print(f"\nTotal Images: {total_images}")
    print(f"Total Labels: {total_labels}")
    
    if "person" in classes and "vest" in classes:
        print("\nAudit PASS. Person and Vest classes exist.")
    else:
        print("\nAudit FAIL. Missing required classes.")

if __name__ == "__main__":
    main()
