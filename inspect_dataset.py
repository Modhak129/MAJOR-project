import os
import glob
from collections import Counter

def check_dataset_annotations(yaml_path):
    print("Parsing SH17 labels...")
    label_dir = r"C:\Major Project\Results\federated_industrial_safety\work\dataset\train\labels"
    label_files = glob.glob(os.path.join(label_dir, "*.txt"))
    
    class_counts = Counter()
    helmet_images = 0
    
    for f in label_files:
        has_helmet = False
        with open(f, 'r') as file:
            for line in file:
                cls_id = int(line.strip().split()[0])
                class_counts[cls_id] += 1
                if cls_id == 14:
                    has_helmet = True
        if has_helmet:
            helmet_images += 1
            
    print(f"Total training labels: {len(label_files)}")
    print(f"Helmet instances (Class 14): {class_counts.get(14, 0)}")
    print(f"Person instances (Class 0): {class_counts.get(0, 0)}")
    print(f"Images with at least one helmet: {helmet_images}")

if __name__ == "__main__":
    check_dataset_annotations("")
