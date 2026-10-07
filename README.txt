Project Setup and Run Instructions
==================================

Prerequisites:
- Python 3.10+
- Node.js & npm
- Docker Desktop (Required if managing edge node containers via dashboard)

1. Python Dependencies
----------------------
Create a virtual environment and install the required Python packages:
  python -m venv venv
  venv\Scripts\activate      (On Windows)
  source venv/bin/activate   (On Mac/Linux)

Install the project dependencies:
  pip install -r requirements.txt
  (or: pip install -r dashboard/requirements.txt)

2. Node.js Dependencies (React Dashboard)
-----------------------------------------
The `node_modules` folder is excluded to keep the package clean.
To install frontend dependencies:
  cd dashboard/frontend-react
  npm install
  cd ../..

3. Trained Model Weights (Included!)
------------------------------------
All required trained PPE detection models are already bundled inside this package:
- `edge/weights/clean_sh17_yolov8n_640_best.pt` / `.onnx` (Nano-tier worker/PPE detector)
- `edge/weights/sh17_yolov8s_fine_best.pt` (Small-tier high-accuracy model)
- `edge/weights/hardhat_helmet_yolov8n_640_best.pt` / `.onnx` (Stage-2 Hardhat/Helmet detector)
- `edge/weights/stage2_vest_yolov8n_640_best.pt` / `.onnx` (Stage-2 Safety Vest detector)
- `models/yolov8n.pt`, `models/yolov8s.pt`, `models/yolov8n.onnx` (Base benchmark models)

No manual model weight downloads are needed.

4. Running the Project
----------------------
A. Standard Local Execution (Recommended for quick testing):
   1. FL Server:
      python server/fl_server.py
   2. Edge Client / Inference:
      python edge/fl_client.py
   3. Flask Dashboard Backend:
      cd dashboard
      python app.py
   4. React Dashboard Frontend:
      cd dashboard/frontend-react
      npm run dev
      (Open http://localhost:5173 in your browser)

B. Docker Execution (For containerized deployment):
   Ensure Docker Desktop is running, then you can use Docker Compose:
      docker-compose up --build
   Or build and run individual edge containers:
      docker build -t edge_node -f edge/Dockerfile .
      docker run --name edge_node_1 -d edge_node
      docker run --name edge_node_2 -d edge_node
