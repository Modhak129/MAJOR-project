

from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
import docker
import json
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///project.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
try:
    docker_client = docker.from_env()
except Exception as e:
    print(f"[Warning] Docker SDK connection failed: {e}")
    docker_client = None

from models import init_db, db, AlertLog, FLRoundMetric, DeviceProfile
init_db(app)

alerts = []
metrics = {
    "round": 0,
    "map": 0.0,
    "loss": 0.0,
    "clients": {},
    "savings_bytes": 0
}
metrics_history = []
geofence_polygon = []

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    return jsonify(alerts)

@app.route('/api/server/events', methods=['POST'])
def receive_server_events():
    data = request.json
    socketio.emit('fl_server_event', data)
    return jsonify({"status": "success"})

@app.route('/api/alerts', methods=['POST'])
def receive_alert():
    data = request.json
    alerts.insert(0, data)
    if len(alerts) > 50:
        alerts.pop()
        
    # Write to database
    with app.app_context():
        import json
        details_val = data.get('details')
        if isinstance(details_val, dict) or isinstance(details_val, list):
            details_str = json.dumps(details_val)
        else:
            details_str = str(details_val) if details_val else None

        new_alert = AlertLog(
            node_id=data.get('client_id'),
            timestamp=data.get('timestamp'),
            violation_type=data.get('type'),
            details=details_str,
            confidence=data.get('confidence'),
            frame_ref=data.get('frame_ref')
        )
        db.session.add(new_alert)
        db.session.commit()
        
    socketio.emit('new_alert', data)
    return jsonify({"status": "success"})

@app.route('/api/db/alerts', methods=['GET'])
def get_db_alerts():
    with app.app_context():
        logs = AlertLog.query.order_by(AlertLog.id.desc()).limit(100).all()
        return jsonify([{"id": l.id, "node_id": l.node_id, "type": l.violation_type, "confidence": l.confidence, "time": l.timestamp} for l in logs])

@app.route('/api/metrics', methods=['POST'])
def update_metrics():
    global metrics, metrics_history
    data = request.json
    metrics.update(data)
    
    # Store history for graphs
    metrics_history.append({
        "round": metrics["round"],
        "map": metrics.get("map", 0),
        "loss": metrics.get("loss", 0)
    })
    if len(metrics_history) > 100:
        metrics_history.pop(0)
        
    # Write to DB
    with app.app_context():
        new_metric = FLRoundMetric(
            session_id="session_" + str(datetime.utcnow().date()),
            round_number=metrics["round"],
            global_loss=metrics.get("loss", 0),
            global_map=metrics.get("map", 0),
            client_metrics_json=json.dumps(metrics.get("clients", {})),
            bytes_saved_pct=metrics.get("savings_pct", 0),
            bytes_transmitted=metrics.get("savings_bytes", 0)
        )
        db.session.add(new_metric)
        db.session.commit()
        
    socketio.emit('metrics_update', metrics)
    socketio.emit('metrics_history_update', metrics_history)
    return jsonify({"status": "success"})

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    return jsonify(metrics)

@app.route('/api/metrics/history', methods=['GET'])
def get_metrics_history():
    return jsonify(metrics_history)

@app.route('/api/db/fl-rounds', methods=['GET'])
def get_db_fl_rounds():
    with app.app_context():
        logs = FLRoundMetric.query.order_by(FLRoundMetric.id.desc()).limit(100).all()
        return jsonify([{"round": l.round_number, "map": l.global_map, "loss": l.global_loss, "saved_pct": l.bytes_saved_pct} for l in logs])

@app.route('/api/db/devices', methods=['GET'])
def get_db_devices():
    with app.app_context():
        devices = DeviceProfile.query.all()
        return jsonify([{
            "node_id": d.node_id, 
            "status": d.status, 
            "ip_address": d.ip_address,
            "cpu_cores": d.cpu_cores,
            "ram_mb": d.ram_mb,
            "simulated_vram_mb": d.simulated_vram_mb, 
            "gpu_name": d.gpu_name,
            "assigned_tier": d.assigned_tier,
            "model_format": d.model_format,
            "input_resolution": d.input_resolution,
            "current_inference_fps": d.current_inference_fps,
            "last_training_upload_time": d.last_training_upload_time.isoformat() if d.last_training_upload_time else None,
            "latest_model_received_time": d.latest_model_received_time.isoformat() if d.latest_model_received_time else None,
            "total_rounds_participated": d.total_rounds_participated,
            "last_round_loss": d.last_round_loss,
            "last_round_sparsification_ratio": d.last_round_sparsification_ratio
        } for d in devices])

@app.route('/api/telemetry', methods=['POST'])
def update_telemetry():
    data = request.json
    node_id = data.get("node_id")
    if not node_id: return jsonify({"error": "node_id required"}), 400
    
    with app.app_context():
        device = DeviceProfile.query.get(node_id)
        if not device:
            device = DeviceProfile(node_id=node_id, simulated_vram_mb=data.get('simulated_vram_mb', 2048), cpu_cores=data.get('cpu_cores', 2))
            db.session.add(device)
        
        for key, value in data.items():
            if hasattr(device, key) and key != "node_id":
                if key in ["last_training_upload_time", "latest_model_received_time"] and value:
                    try:
                        setattr(device, key, datetime.fromisoformat(value.replace("Z", "")))
                    except:
                        pass
                else:
                    setattr(device, key, value)
        
        db.session.commit()
        # Broadcast to frontend
        socketio.emit('node_telemetry_update', {
            "node_id": device.node_id,
            "status": device.status,
            "ip_address": device.ip_address,
            "cpu_cores": device.cpu_cores,
            "ram_mb": device.ram_mb,
            "simulated_vram_mb": device.simulated_vram_mb,
            "gpu_name": device.gpu_name,
            "assigned_tier": device.assigned_tier,
            "model_format": device.model_format,
            "input_resolution": device.input_resolution,
            "current_inference_fps": device.current_inference_fps,
            "last_training_upload_time": device.last_training_upload_time.isoformat() if device.last_training_upload_time else None,
            "latest_model_received_time": device.latest_model_received_time.isoformat() if device.latest_model_received_time else None,
            "total_rounds_participated": device.total_rounds_participated,
            "last_round_loss": device.last_round_loss,
            "last_round_sparsification_ratio": device.last_round_sparsification_ratio
        })
    return jsonify({"status": "success"})

@app.route('/api/clients', methods=['POST'])
def update_client():
    data = request.json
    client_id = data.get("client_id")
    metrics["clients"][client_id] = data
    socketio.emit('metrics_update', metrics)
    return jsonify({"status": "success"})

# --- Docker SDK Integration ---
@app.route('/api/containers', methods=['GET'])
def get_containers():
    if not docker_client: return jsonify([])
    containers = docker_client.containers.list(all=True)
    res = []
    for c in containers:
        if "edge_node" in c.name or "project-2-edge_node" in c.name:
            res.append({
                "id": c.id,
                "name": c.name,
                "status": c.status
            })
    return jsonify(res)

@app.route('/api/containers/<container_id>/<action>', methods=['POST'])
def container_action(container_id, action):
    try:
        containers = docker_client.containers.list(all=True)
        target = None
        for c in containers:
            if c.id == container_id or c.name == container_id or container_id in c.name:
                target = c
                break
                
        if not target:
            return jsonify({"error": f"No such container containing {container_id}"}), 404
            
        if action == 'start':
            target.start()
        elif action == 'stop':
            target.stop()
        elif action == 'restart':
            target.restart()
        return jsonify({"status": f"Successfully {action}ed container {target.name}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Edge Node Control ---
@app.route('/api/containers/<node_id>/mode', methods=['POST'])
def set_edge_mode(node_id):
    data = request.json
    mode = data.get("mode")
    client_id = "node_constrained" if node_id == "edge_node_1" else "node_capable" if node_id == "edge_node_2" else node_id
    socketio.emit('edge_control', {"client_id": client_id, "command": "mode_switch", "mode": mode})
    return jsonify({"status": f"Switched {client_id} to {mode}"})

@app.route('/api/broadcast', methods=['POST'])
def broadcast_global_model():
    data = request.json
    socketio.emit('global_model_broadcast', data)
    return jsonify({"status": "broadcasted"})

@app.route('/api/edge/<client_id>/control', methods=['POST'])
def control_edge(client_id):
    data = request.json
    # Broadcast control command to edge nodes via WebSocket
    socketio.emit('edge_control', {"client_id": client_id, "command": data.get("command")})
    return jsonify({"status": "Command sent"})

# --- Geofencing ---
@app.route('/api/geofence', methods=['POST'])
def set_geofence():
    global geofence_polygon
    data = request.json
    geofence_polygon = data.get("polygon", [])
    # Broadcast to all edge nodes
    socketio.emit('update_geofence', {"polygon": geofence_polygon})
    return jsonify({"status": "Geofence updated"})

@app.route('/api/geofence', methods=['GET'])
def get_geofence():
    return jsonify({"polygon": geofence_polygon})

# --- WebSocket Handlers ---
@socketio.on('connect')
def handle_connect():
    print("Client connected")
    # Send current geofence to newly connected clients
    emit('update_geofence', {"polygon": geofence_polygon})

@socketio.on('frontend_frame')
def handle_frontend_frame(data):
    # Route frame to specified edge node for processing
    # data is a dict: { image: base64, target_node: "node_constrained" }
    emit('process_frame', data, broadcast=True, include_self=False)

@socketio.on('processed_frame_result')
def handle_processed_frame(data):
    print("Received processed_frame_result from edge node!", flush=True)
    # Route processed frame back to frontend
    emit('processed_frame_result', data, broadcast=True, include_self=False)

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)
