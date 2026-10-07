import cv2
import os

videos = ["JapanPPE.mp4", "hardhat.mp4", "indianworkers.mp4"]
base_path = r"data\raw_dataset\source_files\source_files"

for v in videos:
    cap = cv2.VideoCapture(os.path.join(base_path, v))
    if not cap.isOpened():
        print(f"{v}: Failed to open.")
        continue
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frames / fps if fps else 0
    print(f"{v}: {w}x{h}, {fps:.2f} fps, {frames} frames, {duration:.2f} seconds")
    cap.release()
