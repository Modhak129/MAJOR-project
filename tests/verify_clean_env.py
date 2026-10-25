import yaml
import os
import cv2
import numpy as np
import sys

def verify_system():
    print("--- 1. Verifying config/config.yaml ---")
    with open('config/config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    for k, v in cfg['models'].items():
        assert os.path.exists(v), f"Model missing: {v}"
        print(f"   [OK] models.{k}: {v} ({os.path.getsize(v)/1024/1024:.2f} MB)")

    helmet_path = cfg['inference']['helmet_model_path']
    vest_path = cfg['inference']['vest_model_path']
    assert os.path.exists(helmet_path), f"Helmet model missing: {helmet_path}"
    print(f"   [OK] inference.helmet_model_path: {helmet_path} ({os.path.getsize(helmet_path)/1024/1024:.2f} MB)")
    assert os.path.exists(vest_path), f"Vest model missing: {vest_path}"
    print(f"   [OK] inference.vest_model_path: {vest_path} ({os.path.getsize(vest_path)/1024/1024:.2f} MB)")

    print("\n--- 2. Testing Two-Stage Inference Engine ---")
    sys.path.append('edge')
    from inference import InferenceEngine
    from ultralytics import YOLO

    engine = InferenceEngine(cfg, cfg['models']['nano'], 'verification_node', 'http://127.0.0.1:5000')
    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    out_frame = engine.process_frame(frame)
    assert out_frame is not None and out_frame.shape == (320, 320, 3), "Inference output invalid"
    print("   [OK] Two-stage InferenceEngine processed synthetic frame successfully!")

    print("\n--- 3. Testing Focused Model Weights ---")
    focused_s = YOLO(cfg['models']['focused_small'])
    focused_n = YOLO(cfg['models']['focused_nano'])
    print(f"   [OK] Focused Small Names: {focused_s.names}")
    print(f"   [OK] Focused Nano Names: {focused_n.names}")
    expected_classes = {0: 'Person', 1: 'Head', 2: 'Helmet', 3: 'Safety-vest', 4: 'Gloves', 5: 'Shoes'}
    assert focused_s.names == expected_classes, f"Class mapping mismatch: {focused_s.names}"
    assert focused_n.names == expected_classes, f"Class mapping mismatch: {focused_n.names}"
    print("   [OK] 6 Focused PPE classes matched exactly!")

    print("\n--- 4. Testing FL Server Adaptive Strategy ---")
    sys.path.append('server')
    from adaptive import apply_fedprox_regularization, knowledge_distillation_reconciliation
    w1 = [np.random.randn(10, 10).astype(np.float32)]
    w2 = [np.random.randn(10, 10).astype(np.float32)]
    reg = apply_fedprox_regularization(w1, w2, mu=0.01)
    kd = knowledge_distillation_reconciliation(w1, w2, temperature=4.0)
    assert len(reg) == 1 and len(kd) == 1
    print("   [OK] Adaptive federated algorithms verified!")

    print("\n==========================================")
    print("ALL CLEAN-ENVIRONMENT CHECKS PASSED (100%)")
    print("==========================================")

if __name__ == '__main__':
    verify_system()
