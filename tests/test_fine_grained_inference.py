import pytest
import sys
sys.path.append("/app/edge")
from unittest.mock import MagicMock
from inference import InferenceEngine
from shapely.geometry import Polygon

@pytest.fixture
def mock_inference_engine():
    # Mocking YOLO and AlertManager
    class MockYOLO:
        def track(self, frame, **kwargs):
            return [MagicMock()]
    
    import inference
    inference.YOLO = MagicMock(return_value=MockYOLO())
    
    config = {
        'inference': {
            'min_confidence': 0.5,
            'alert_cooldown_sec': 10,
            'input_resolution': 640
        }
    }
    
    engine = InferenceEngine(config, "dummy.onnx", "test_client", "http://dummy")
    engine.model = MockYOLO()
    engine.alert_manager = MagicMock()
    return engine

def test_missing_ppe_logic(mock_inference_engine):
    """
    Test that if a person bounding box does not contain helmet/jacket/gloves, 
    the proper alert is generated.
    """
    engine = mock_inference_engine
    
    # Create mock boxes
    class MockBox:
        def __init__(self, cls_id, xyxy):
            self.conf = [0.9]
            self.cls = [cls_id]
            self.id = [1]
            self.xywhn = [[0, 0, 0, 0]]
            self.xyxy = [xyxy]
            
    # Person box from 0,0 to 100,100
    person_box = MockBox(0, [0, 0, 100, 100])
    
    # Helmet box contained entirely in person box
    helmet_box = MockBox(1, [10, 10, 30, 30])
    
    # Jacket box entirely contained
    jacket_box = MockBox(2, [20, 40, 80, 80])
    
    # Missing gloves (class 3)
    
    mock_results = MagicMock()
    mock_results.boxes = [person_box, helmet_box, jacket_box]
    mock_results.plot.return_value = "dummy_frame"
    engine.model.track = MagicMock(return_value=[mock_results])
    
    # Mock saliency map
    import inference
    inference.generate_saliency_map = MagicMock(return_value="dummy_xai")
    
    import cv2
    cv2.imwrite = MagicMock()
    engine.process_frame("dummy_frame")
    
    # Check if AlertManager was called with missing gloves
    calls = engine.alert_manager.send_alert.call_args_list
    assert len(calls) == 1
    args, kwargs = calls[0]
    
    assert args[1] == "Missing PPE"
    assert "gloves" in kwargs['details']['missing_items']
    assert "helmet" not in kwargs['details']['missing_items']

def test_geofence_limb_intrusion(mock_inference_engine):
    """
    Test that if a limb (hand/leg) enters the geofence, 
    an intrusion alert is generated.
    """
    engine = mock_inference_engine
    
    # Set restricted polygon
    engine.geofence_polygon = [[50, 50], [150, 50], [150, 150], [50, 150]]
    
    class MockBox:
        def __init__(self, cls_id, xyxy):
            self.conf = [0.9]
            self.cls = [cls_id]
            self.id = [1]
            self.xywhn = [[0, 0, 0, 0]]
            self.xyxy = [xyxy]
            
    # Person box completely outside the zone, but hand reaches in
    person_box = MockBox(0, [0, 0, 60, 60])
    
    # But a hand reaches in (crosses into 50,50 boundary)
    hand_box = MockBox(4, [45, 45, 60, 60])
    
    mock_results = MagicMock()
    mock_results.boxes = [person_box, hand_box]
    mock_results.plot.return_value = "dummy_frame"
    engine.model.track = MagicMock(return_value=[mock_results])
    
    import inference
    inference.generate_saliency_map = MagicMock(return_value="dummy_xai")
    
    import cv2
    cv2.imwrite = MagicMock()
    cv2.polylines = MagicMock()
    engine.process_frame("dummy_frame")
    
    calls = engine.alert_manager.send_alert.call_args_list
    # Note: Person is missing PPE as well, so there will be 2 alerts
    assert len(calls) == 2
    
    geofence_call = [c for c in calls if c[0][1] == "Zone Intrusion"][0]
    args, kwargs = geofence_call
    
    assert args[1] == "Zone Intrusion"
    assert "Hand" in kwargs['details']
