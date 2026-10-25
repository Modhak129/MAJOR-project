# Industrial PPE Safety Monitoring & Federated Edge AI System

An end-to-end industrial computer vision and federated edge AI system engineered for real-time Personal Protective Equipment (PPE) compliance detection, worker tracking, geofence intrusion monitoring, and adaptive federated model aggregation.

---

## Overview

Workplace safety compliance in construction, manufacturing, and industrial facilities is critical to preventing injuries and fatalities. The **Industrial PPE Safety Monitoring System** provides an automated, computer vision-driven surveillance framework that operates on edge devices and central dashboard servers.

The system performs:
- High-speed worker detection and multi-object tracking.
- Fine-grained PPE compliance verification (Hardhat/Helmet, Safety Vest, Gloves, Footwear/Shoes, Head).
- Restricted hazardous zone intrusion detection via user-configurable geofences.
- Dynamic telemetry, alert dispatches, and explainable AI (XAI) visual auditing.
- Adaptive tier Federated Learning (Flower) with knowledge distillation and communication sparsification between edge nodes and central aggregation servers.

---

## Key Features

- **Multi-Class Detection & PPE Verification**:
  - Worker identification (`Person`, `Head`)
  - Head protection (`Helmet` / Hardhat)
  - Torso protection (`Safety-vest` / High-visibility jacket)
  - Hand protection (`Gloves`)
  - Foot protection (`Shoes` / Work boots)
- **Two-Stage Detection Architecture**:
  - **Stage 1**: Detects workers (`Person`) and tracks individual IDs frame-by-frame across CCTV streams using ByteTrack (`bytetrack.yaml`).
  - **Stage 2**: Automatically crops worker bounding regions with dynamic spatial expansions (`helmet_crop_expansion: 0.20`, `vest_crop_expansion: 0.20`) and executes specialized Stage-2 detection heads for hardhats and high-visibility vests.
- **ByteTrack Multi-Worker Tracking**:
  - Assigns and persists persistent Worker IDs across frames.
  - Automatically resets tracker state upon video source restart or file change.
- **Geofence Intrusion Detection**:
  - Interactive polygon drawing on live camera streams via the React web UI.
  - Polygon-box intersection checking using Shapely geometry engine to immediately flag workers entering hazardous perimeter boundaries.
- **Alert Dispatch & Rate Limiting**:
  - Immediate alert notification via WebSocket (`Socket.IO`) and persistent SQLite logging.
  - Configurable alert cooldown (`alert_cooldown_sec: 10`) to prevent duplicate notification spam for the same worker.
- **Explainable AI (XAI) Heatmap Integration**:
  - Saliency map generation for model attention attribution on flagged violation frames.
- **Video Upload & Interactive Demo Mode**:
  - Live webcam streaming or CCTV surveillance video file upload (`.mp4`, `.avi`, `.mov`).
  - Web UI playback controls: Play, Pause, Replay, and Safe Video Removal.
- **Adaptive Tier Federated Learning**:
  - Hardware-aware dynamic tier assignment (Nano tier for resource-constrained edge devices, Small tier for capable GPU nodes).
  - Top-$k$ parameter sparsification (`sparsification_k: 0.01`) to cut network transmission overhead by ~99%.
  - Central server FedAvg aggregation with knowledge distillation (KD) parameter reconciliation.

---

## System Architecture

```
                 +-------------------------------------------------------+
                 |              CCTV / Video Upload / Webcam             |
                 +-------------------------------------------------------+
                                             |
                                             v
                 +-------------------------------------------------------+
                 |                Stage 1: Person Tracker                |
                 |     (YOLOv8 Detection + ByteTrack ID Association)     |
                 +-------------------------------------------------------+
                                             |
                     +-----------------------+-----------------------+
                     |                                               |
                     v                                               v
        +-------------------------+                     +-------------------------+
        |   Head / Helmet Crop    |                     |    Torso / Vest Crop    |
        |  (Expansion Ratio: 0.20)|                     |  (Expansion Ratio: 0.20)|
        +-------------------------+                     +-------------------------+
                     |                                               |
                     v                                               v
        +-------------------------+                     +-------------------------+
        |  Stage 2: Helmet Model  |                     |   Stage 2: Vest Model   |
        | (hardhat_helmet_yolov8n)|                     | (stage2_vest_yolov8n)   |
        +-------------------------+                     +-------------------------+
                     |                                               |
                     +-----------------------+-----------------------+
                                             |
                                             v
                 +-------------------------------------------------------+
                 |       Association & Compliance Evaluation Engine      |
                 | - Validates worker-to-PPE containment (>50% IoU)      |
                 | - Checks Geofence Polygon boundary intersection       |
                 +-------------------------------------------------------+
                                             |
                                             v
                 +-------------------------------------------------------+
                 |          Alert Manager & WebSocket Dispatch           |
                 | - Dispatches violations to Flask/Socket.IO backend    |
                 | - Visualizes bounded boxes on React Dashboard         |
                 +-------------------------------------------------------+
```

---

## Model Information

The repository includes trained YOLOv8 model weights under `edge/weights/` and `models/`. All weights are stored directly in the repository and are ready to run out of the box.

### Model Catalog

| Model Name | Architecture | Classes | Input Size | File Path | Application Role | Approx Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 2 Helmet Detector** | YOLOv8n | 1 (`Helmet`) | 640×640 | `edge/weights/hardhat_helmet_yolov8n_640_best.pt` | **Production**: High-accuracy Stage-2 worker crop helmet validation | 5.96 MB |
| **Stage 2 Vest Detector** | YOLOv8n | 1 (`safety_vest`) | 640×640 | `edge/weights/stage2_vest_yolov8n_640_best.pt` | **Production**: High-accuracy Stage-2 worker crop safety vest validation | 5.92 MB |
| **SH17 Focused PPE (Small)** | YOLOv8s | 6 (Focused) | 640×640 | `edge/weights/sh17_focused_yolov8s_640_best.pt` | **Trained Focused Model**: Dedicated 6-class comprehensive industrial detector | 21.46 MB |
| **SH17 Focused PPE (Nano)** | YOLOv8n | 6 (Focused) | 640×640 | `edge/weights/sh17_focused_yolov8n_640_best.pt` | **Trained Focused Model**: Lightweight 6-class model for edge devices | 5.94 MB |
| **SH17 Clean PPE (Nano)** | YOLOv8n | 17 (Full SH17) | 640×640 | `edge/weights/clean_sh17_yolov8n_640_best.pt` | **Production Stage 1**: Base tracking & multi-class worker detection | 5.97 MB |
| **SH17 Clean PPE (Small)** | YOLOv8s | 17 (Full SH17) | 640×640 | `edge/weights/sh17_yolov8s_fine_best.pt` | **Production Capable Tier**: High-capacity FL client detector | 21.47 MB |
| **Roboflow PPE Detector** | YOLOv8n | 14 (Roboflow) | 640×640 | `edge/weights/roboflow_yolov8n_640_best.pt` | **Benchmark**: Cross-domain comparative evaluation | 5.98 MB |
| **Base YOLOv8n** | YOLOv8n | 10 (CSS) | 640×640 | `models/yolov8n.pt` | **Baseline**: Standard baseline model | 5.97 MB |
| **Base YOLOv8s** | YOLOv8s | 80 (COCO) | 640×640 | `models/yolov8s.pt` | **Baseline**: Standard baseline model | 21.54 MB |

### Focused Model Classes (6 Classes)
```
0: Person
1: Head
2: Helmet
3: Safety-vest
4: Gloves
5: Shoes
```

### Full SH17 Model Classes (17 Classes)
```
0: Person             5: Face-guard         10: Foot           15: Medical-suit
1: Head               6: Ear                11: Shoes          16: Safety-suit
2: Face               7: Earmuffs           12: Safety-vest
3: Glasses            8: Hands              13: Tools
4: Face-mask-medical  9: Gloves             14: Helmet
```

---

## Project Structure

```
Project-2/
├── config/
│   └── config.yaml                   # Core application & model configuration
├── dashboard/
│   ├── app.py                        # Flask backend API & Socket.IO telemetry server
│   ├── models.py                     # SQLAlchemy database schemas (Alerts, Metrics, Devices)
│   ├── requirements.txt              # Backend-specific Python requirements
│   ├── Dockerfile                    # Container definition for dashboard service
│   ├── templates/
│   │   └── index.html                # Fallback HTML template
│   └── frontend-react/               # Modern React + Vite + Tailwind CSS dashboard
│       ├── src/
│       │   ├── App.jsx               # Navigation & route management
│       │   ├── pages/
│       │   │   ├── LiveMonitoring.jsx# Real-time CCTV analysis & interactive demo player
│       │   │   ├── LiveFeed.jsx      # Multi-camera grid view
│       │   │   ├── DeviceControl.jsx # Node state management & edge mode switches
│       │   │   ├── Geofence.jsx      # Hazard zone polygon manager
│       │   │   └── ...
│       │   └── components/           # UI components & alert feeds
│       ├── package.json              # Frontend npm dependencies
│       ├── vite.config.js            # Vite build configuration
│       └── dist/                     # Pre-compiled static frontend distribution
├── edge/
│   ├── fl_client.py                  # Federated learning edge client & inference coordinator
│   ├── inference.py                  # Real-time two-stage inference & ByteTrack engine
│   ├── alerts.py                     # Cooldown-aware HTTP alert dispatcher
│   ├── xai_stub.py                   # Saliency map attribution generator
│   ├── Dockerfile                    # Container definition for edge nodes
│   └── weights/                      # Trained YOLOv8 models & ONNX exports
├── models/                           # Base reference models
├── server/
│   ├── fl_server.py                  # Federated Learning server (Flower FedAvg)
│   ├── adaptive.py                   # Sparsification & knowledge distillation algorithms
│   └── Dockerfile                    # Container definition for FL server
├── data/
│   ├── partition.py                  # Dataset partitioner for FL experiments
│   └── uploads/
│       └── .gitkeep                  # Storage directory for video uploads
├── tests/                            # Automated test suite
│   ├── test_adaptive.py
│   ├── test_alerts.py
│   ├── test_dataset.py
│   ├── test_fedavg.py
│   ├── test_fine_grained_inference.py
│   └── test_inference.py
├── docker-compose.yml                # Multi-container orchestration (Server + Dashboard + Edge)
├── requirements.txt                  # Complete project Python dependencies
├── .env.example                      # Environment template file
├── .gitignore                        # Git exclusion rules
└── README.md                         # Project documentation
```

---

## Hardware & System Requirements

- **Operating System**: Windows 10/11, Ubuntu 20.04/22.04 LTS, or macOS (Apple Silicon supported).
- **Python Version**: Python 3.10, 3.11, or 3.12.
- **Node.js**: Node.js v18.x or v20.x and npm v9+.
- **GPU (Optional but Recommended)**:
  - NVIDIA GPU with CUDA 11.8 or 12.x support (e.g., RTX 3060/4060 or higher).
  - CPU inference is supported automatically via PyTorch fallback.
- **RAM**: Minimum 8 GB (16 GB recommended).
- **Disk Space**: ~2 GB for dependencies and model weights (no datasets required).

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Modhak129/MAJOR-project.git
cd MAJOR-project
```

### 2. Set Up Python Environment
Create and activate an isolated virtual environment:
```bash
# On Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# On Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install the required Python packages:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on PyTorch with CUDA**: If using an NVIDIA GPU, install the appropriate CUDA PyTorch wheel for your system:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 3. Set Up React Frontend
```bash
cd dashboard/frontend-react
npm install
npm run build
cd ../..
```

### 4. Configure Environment Variables
Copy the example environment template:
```bash
# On Linux / macOS:
cp .env.example .env

# On Windows (PowerShell):
Copy-Item .env.example .env
```

---

## Running the Application

### Option A: Local Native Execution (Recommended for Development)

Open separate terminal windows for each service (ensure your virtual environment is active in each):

#### Terminal 1: Federated Learning Server
```bash
python server/fl_server.py
```
*Starts the Flower gRPC aggregation server on port 8080.*

#### Terminal 2: Flask Dashboard & Socket.IO Backend
```bash
python dashboard/app.py
```
*Starts the REST API and WebSocket server on `http://localhost:5000`.*

#### Terminal 3: React Web Frontend
```bash
cd dashboard/frontend-react
npm run dev
```
*Starts the Vite dev server at `http://localhost:5173`.*

#### Terminal 4: Edge AI Client / Inference Engine
```bash
# Set environment variables for the node (Windows PowerShell):
$env:CLIENT_ID="node_constrained"
$env:SERVER_ADDRESS="127.0.0.1:8080"
$env:DASHBOARD_URL="http://127.0.0.1:5000"
$env:SIMULATED_VRAM_MB="2048"

python edge/fl_client.py
```
*(On Linux/macOS, use `export CLIENT_ID="node_constrained"` etc.)*

---

### Option B: Docker Compose Deployment

If Docker Desktop is installed and running:
```bash
docker-compose up --build
```
This automatically starts:
- `fl_server`: Port 8080
- `dashboard`: Port 5000
- `edge_node_1`: Constrained edge client
- `edge_node_2`: Capable edge client

Open `http://localhost:5000` in your browser.

---

## Interactive Demo Mode & Video Upload

1. Navigate to `http://localhost:5173` (or `http://localhost:5000` in production) and open the **Live Monitoring** tab.
2. Select your target edge node (`edge_node_1` or `edge_node_2`).
3. Under **Video Source**, select **Upload Video**.
4. Choose an MP4/AVI surveillance recording. The file will upload to `data/uploads/`.
5. Click **Start Demo Analysis** to initiate two-stage worker tracking and PPE verification.
6. Use **Pause**, **Play**, **Replay**, or **Remove Video** to manage the playback lifecycle.

---

## Configuration Reference (`config/config.yaml`)

```yaml
inference:
  ppe_detector_mode: "two_stage"      # Options: 'two_stage' or 'sh17'
  helmet_model_path: "edge/weights/hardhat_helmet_yolov8n_640_best.pt"
  vest_model_path: "edge/weights/stage2_vest_yolov8n_640_best.pt"
  helmet_conf: 0.40                   # Confidence threshold for hardhat detection
  helmet_crop_expansion: 0.20         # 20% spatial padding on head crop
  vest_crop_expansion: 0.20           # 20% spatial padding on torso crop
  input_resolution: 320               # Inference image dimensions
  min_confidence: 0.45                # General confidence cutoff
  vest_confidence: 0.35               # Specific confidence threshold for vest
  alert_cooldown_sec: 10              # Minimum seconds between duplicate alerts

models:
  nano: "edge/weights/clean_sh17_yolov8n_640_best.pt"
  small: "edge/weights/sh17_yolov8s_fine_best.pt"
  focused_small: "edge/weights/sh17_focused_yolov8s_640_best.pt"
  focused_nano: "edge/weights/sh17_focused_yolov8n_640_best.pt"

server:
  grpc_port: 8080
  dashboard_port: 5000
  checkpoint_dir: "experiments/checkpoints"
```

---

## Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **CUDA out of memory** | High batch size or concurrent model allocations | In `config/config.yaml`, lower `input_resolution` to `320` or set device to `'cpu'`. |
| **Model file not found** | Relative path resolution mismatch | Ensure scripts are executed from the project root (`MAJOR-project/`). Verify files exist in `edge/weights/`. |
| **`npm.ps1 cannot be loaded`** | Windows PowerShell script execution policy | Use `npm.cmd run build` or run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`. |
| **Socket.IO connection refused** | Backend server not yet running | Start `dashboard/app.py` before launching `edge/fl_client.py` or the React frontend. |
| **Port 5000 already in use** | Another process is using port 5000 | Terminate the occupying process or change `dashboard_port` in `config/config.yaml` and `.env`. |
| **Docker volume permission error** | Docker Desktop lacks file sharing access | Under Docker Desktop Settings -> Resources -> File Sharing, ensure the repository directory is checked. |

---

## License

This project is developed for educational and industrial research purposes. All YOLOv8 dependencies are subject to their respective upstream licenses.
