Project Setup and Run Instructions

Since this archive excludes heavy dependencies and models to keep the file size small, you will need to download and install a few things before running the project.

Prerequisites:
- Python 3.10+
- Node.js & npm
- Docker Desktop (Required for the dashboard to manage edge containers)

1. Python Dependencies (for Dashboard)
--------------------------------------
Create a virtual environment and install the required Python packages:
  python -m venv venv
  venv\Scripts\activate      (On Windows)
  source venv/bin/activate   (On Mac/Linux)

Then, install the requirements:
  pip install -r dashboard/requirements.txt

2. Node.js Dependencies (React Dashboard)
-----------------------------------------
The `node_modules` folder was excluded. To install the frontend dependencies, navigate to the frontend folder and run:
  cd dashboard/frontend-react
  npm install

3. YOLO Models
--------------
The pre-trained model files were excluded due to size. You need to download the following models and place them in the `models/` folder:
- yolov8n.pt
- yolov8s.pt
- yolov8n.onnx
(You can download these from the official Ultralytics YOLOv8 releases).

4. Running the Project with Docker Integration
----------------------------------------------
The dashboard uses the Docker Python SDK to manage the edge nodes. Therefore, Docker Desktop MUST be running on your machine.

A. Run the FL Server (Local):
   In Terminal 1 (with virtual environment activated):
   `python server/fl_server.py`

B. Build and Run the Edge Node Containers:
   In Terminal 2, build the edge node image from the root directory. 
   *(Note: Ensure you have a requirements.txt in the root directory before building, or copy dashboard/requirements.txt if they share dependencies)*:
   `copy dashboard\requirements.txt .`
   `docker build -t edge_node -f edge/Dockerfile .`
   
   Then run the edge clients as background containers so the dashboard can control them:
   `docker run --name edge_node_1 -d edge_node`
   `docker run --name edge_node_2 -d edge_node`

C. Run the Flask Dashboard Backend:
   In Terminal 3 (with virtual environment activated):
   `cd dashboard`
   `python app.py`

D. Run the React Dashboard Frontend:
   In Terminal 4:
   `cd dashboard/frontend-react`
   `npm run dev`
   (Open the local URL provided, usually http://localhost:5173).

Now, the dashboard will connect to Docker Desktop and allow you to start/stop `edge_node_1` and `edge_node_2` directly from the web interface!
