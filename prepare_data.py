import os

def rewrite_file(input_txt, output_txt, prefix):
    with open(input_txt, 'r') as f:
        lines = f.read().splitlines()
        
    with open(output_txt, 'w') as f:
        for line in lines:
            if line.strip():
                # Some YOLO versions want absolute paths
                full_path = os.path.join(prefix, line.strip())
                f.write(f"{full_path}\n")

if __name__ == "__main__":
    sh17_dir = r"C:\Major Project\Results\datasets\SH17"
    img_dir = os.path.join(sh17_dir, "images")
    
    rewrite_file(os.path.join(sh17_dir, "train_files.txt"), "sh17_train.txt", img_dir)
    rewrite_file(os.path.join(sh17_dir, "val_files.txt"), "sh17_val.txt", img_dir)
    print("Created absolute path text files.")
