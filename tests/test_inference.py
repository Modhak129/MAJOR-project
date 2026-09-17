import pytest
import numpy as np
import sys
import os
import cv2
from unittest.mock import patch, MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../edge')))
from inference import InferenceEngine

@patch('inference.YOLO')
def test_inference_engine_loads(mock_yolo):
    # Test EIM-01
    config = {
        'inference': {'min_confidence': 0.45, 'alert_cooldown_sec': 10}
    }
    
    # Initialize engine
    engine = InferenceEngine(config, "mock_path.pt", "node_1", "http://mock")
    
    # Verify YOLO was instantiated
    mock_yolo.assert_called_once_with("mock_path.pt")

@patch('inference.YOLO')
@patch('inference.AlertManager')
def test_inference_engine_detects(mock_alert, mock_yolo):
    # Test EIM-06
    config = {
        'inference': {'min_confidence': 0.45, 'alert_cooldown_sec': 10}
    }
    
    # Mock YOLO results
    mock_box = MagicMock()
    mock_box.conf = [0.85]
    mock_box.cls = [0] # PPE Violation in our synthetic map
    mock_box.xywh = [[[100, 100, 50, 50]]]
    
    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_result.plot.return_value = np.zeros((320, 320, 3), dtype=np.uint8)
    
    mock_yolo_instance = mock_yolo.return_value
    mock_yolo_instance.return_value = [mock_result]
    
    engine = InferenceEngine(config, "mock_path.pt", "node_1", "http://mock")
    
    # Mock frame
    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    
    # Process
    res = engine.process_frame(frame)
    
    # Verify alert was sent
    engine.alert_manager.send_alert.assert_called_once_with("node_1", "PPE Violation", 0.85)
    
    # Verify min confidence filter
    engine.alert_manager.send_alert.reset_mock()
    mock_box.conf = [0.2] # Below threshold
    engine.process_frame(frame)
    engine.alert_manager.send_alert.assert_not_called()
