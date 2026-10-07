# TWO-STAGE + UI STATE DIAGNOSTIC

## 1. Stage-1 status
Model: clean_sh17_yolov8n_640_best.pt
Loaded and executed in both diagnostic and Docker environments.
Detects Person successfully.
Filters out Safety-vest due to low confidence (0.36 < 0.45 threshold).

## 2. Stage-2 status
Docker Environment: FAILED TO LOAD.
Diagnostic Environment: LOADED AND EXECUTED.

## 3. Stage-2 helmet result
Docker Environment: None (skipped because model was not loaded).
Diagnostic Environment: Detected (confidence 0.81).

## 4. Compliance result
Docker Environment: `missing = ['helmet', 'safety vest']` (Falls back to SH17 which missed helmet, plus vest suppressed).
Diagnostic Environment: `missing = ['safety vest']`.

## 5. Backend alert payload
Docker Payload: `{"type": "Missing PPE", "details": {"missing_items": ["helmet", "safety vest"]}, ...}`

## 6. Socket.IO/API payload
Transmitted successfully via HTTP POST from `alerts.py` to Flask, which emitted via `socketio.emit('new_alert', data)`.

## 7. React received payload
Received exactly as emitted. `missing_items: ["helmet", "safety vest"]`.

## 8. React rendered state
Renders a Badge component for each string in `missing_items`. 
Mapped directly to: "No helmet" and "No safety vest".

## 9. Backend/frontend mismatch
There is NO mismatch between the running Docker backend and the frontend. The mismatch was between your local diagnostic test (Windows) and the live running application (Docker).

## 10. Multiple-process check
YES. The diagnostic ran locally on Windows using `venv_edge_312`. The actual UI is powered by `docker-compose` Linux containers (`project-2-edge_node_1-1`).

## 11. Exact root cause

[x] Wrong configuration loaded by UI process
[x] Multiple backend processes

**Concrete Evidence:**
1. **Invalid Path in Docker:** Your `config.yaml` contained an absolute Windows path for the Stage-2 model: `C:/Major Project/Project-2/runs/detect/edge/weights/training/hardhat_helmet_yolov8n_640/weights/best.pt`.
2. **Missing Volume Map:** The `runs/` directory is not mapped into the Docker containers via `docker-compose.yml`. Thus, the file physically did not exist inside the container.
3. **Silent Failure Fallback:** When `os.path.exists()` failed inside the Linux Docker container, `inference.py` logged `Warning: two_stage mode enabled but helmet model path is invalid!` and set `self.helmet_model = None`. 
4. **Result:** The Stage-2 logic was completely skipped in the live application, and it fell back to SH17 which missed the helmet. The vest was also dropped by the 0.45 threshold. Thus both items were added to the `missing_items` array.

**Action Taken to Fix:**
I have copied the trained Stage-2 model into the `edge/weights/` directory (which is mapped to the Docker container) and changed the `config.yaml` to use the relative path `edge/weights/hardhat_helmet_yolov8n_640_best.pt`. 

You just need to restart the edge node containers to apply the fix.
