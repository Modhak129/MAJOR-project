from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class AlertLog(db.Model):
    __tablename__ = 'alert_log'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    node_id = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.String(50), nullable=False)
    violation_type = db.Column(db.String(50), nullable=False)
    details = db.Column(db.String(500), nullable=True)
    confidence = db.Column(db.Float, nullable=False)
    frame_ref = db.Column(db.String(255), nullable=True)
    xai_heatmap_ref = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class FLRoundMetric(db.Model):
    __tablename__ = 'fl_round_metric'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    session_id = db.Column(db.String(100), nullable=False)
    round_number = db.Column(db.Integer, nullable=False)
    global_loss = db.Column(db.Float, nullable=False)
    global_map = db.Column(db.Float, nullable=False)
    client_metrics_json = db.Column(db.Text, nullable=True)
    bytes_saved_pct = db.Column(db.Float, nullable=True)
    bytes_transmitted = db.Column(db.Integer, nullable=True)
    duration_seconds = db.Column(db.Float, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class DeviceProfile(db.Model):
    __tablename__ = 'device_profile'
    node_id = db.Column(db.String(50), primary_key=True)
    status = db.Column(db.String(20), default="Idle")
    ip_address = db.Column(db.String(50), nullable=True)
    
    # Hardware Profile
    cpu_cores = db.Column(db.Integer, nullable=False)
    ram_mb = db.Column(db.Integer, nullable=True)
    simulated_vram_mb = db.Column(db.Integer, nullable=False)
    gpu_name = db.Column(db.String(100), nullable=True)
    
    # Active Model Metadata
    assigned_tier = db.Column(db.String(20), nullable=True)
    model_format = db.Column(db.String(20), nullable=True)
    input_resolution = db.Column(db.String(20), nullable=True)
    current_inference_fps = db.Column(db.Float, nullable=True)
    
    # Federated Learning Activity
    last_training_upload_time = db.Column(db.DateTime, nullable=True)
    latest_model_received_time = db.Column(db.DateTime, nullable=True)
    total_rounds_participated = db.Column(db.Integer, default=0)
    last_round_loss = db.Column(db.Float, nullable=True)
    last_round_sparsification_ratio = db.Column(db.String(20), nullable=True)
    
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

def init_db(app):
    db.init_app(app)
    with app.app_context():
        db.create_all()
