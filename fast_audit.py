import os
import yaml
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def process_file(file_path):
    with open(file_path, 'r') as f:
        data = f.read()
    counts = defaultdict(int)
    for line in data.splitlines():
        if line.strip():
            cls = int(line.split()[0])
            counts[cls] += 1
    return counts

def fast_audit():
    dataset_dir = Path(r"C:\Major Project\Project-2\data\raw_dataset\Dataset Roboflow")
    with open(dataset_dir / "data.yaml", "r") as f:
        data = yaml.safe_load(f)
    names = data['names']
    nc = data['nc']
    
    for split in ['train', 'valid', 'test']:
        print(f"\n--- Split: {split} ---")
        lbl_dir = dataset_dir / split / 'labels'
        if not lbl_dir.exists():
            print("Not found.")
            continue
        
        files = list(lbl_dir.glob('*.txt'))
        print(f"Total images: {len(files)}")
        
        total_counts = defaultdict(int)
        with ThreadPoolExecutor() as executor:
            for counts in executor.map(process_file, files):
                for k, v in counts.items():
                    total_counts[k] += v
                    
        print("Annotations:")
        for cls_id in sorted(total_counts.keys()):
            print(f"  {names[cls_id]}: {total_counts[cls_id]}")

if __name__ == "__main__":
    fast_audit()
