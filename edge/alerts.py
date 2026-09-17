import time
import requests
import json
import os
from datetime import datetime

class AlertManager:
    def __init__(self, dashboard_url, cooldown=10):
        self.dashboard_url = dashboard_url
        self.cooldown = cooldown
        self.history = {}
        os.makedirs("logs", exist_ok=True)

    def send_alert(self, client_id, alert_type, confidence, worker_id=None, details=None):
        now = time.time()
        if alert_type not in self.history or (now - self.history[alert_type]) > self.cooldown:
            self.history[alert_type] = now
            
            payload = {
                "client_id": client_id,
                "type": alert_type,
                "confidence": confidence,
                "worker_id": worker_id,
                "details": details,
                "timestamp": datetime.utcfromtimestamp(now).isoformat() + "Z"
            }
            
            # Write to local JSONL
            date_str = datetime.utcnow().strftime("%Y-%m-%d")
            log_file = f"logs/alerts_{client_id}_{date_str}.jsonl"
            with open(log_file, "a") as f:
                f.write(json.dumps(payload) + "\n")
                
            try:
                requests.post(f"{self.dashboard_url}/api/alerts", json=payload)
            except Exception as e:
                print(f"Failed to send alert: {e}")
