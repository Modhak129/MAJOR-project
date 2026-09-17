import pytest
import time
from unittest.mock import patch, MagicMock
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../edge')))

from alerts import AlertManager

@pytest.fixture
def alert_manager():
    return AlertManager(dashboard_url="http://mock-dashboard:5000", cooldown=2)

@patch('requests.post')
def test_alert_schema_and_generation(mock_post, alert_manager):
    # Test ANM-01, ANM-02
    alert_manager.send_alert("node_1", "PPE Violation", 0.95)
    
    assert mock_post.called
    args, kwargs = mock_post.call_args
    payload = kwargs['json']
    
    assert payload['client_id'] == "node_1"
    assert payload['type'] == "PPE Violation"
    assert payload['confidence'] == 0.95
    assert 'timestamp' in payload

@patch('requests.post')
def test_alert_cooldown(mock_post, alert_manager):
    # Test ANM-03: Suppress duplicate alerts
    alert_manager.send_alert("node_1", "Fall Detected", 0.88)
    assert mock_post.call_count == 1
    
    # Send again immediately (should be suppressed by cooldown)
    alert_manager.send_alert("node_1", "Fall Detected", 0.90)
    assert mock_post.call_count == 1
    
    # Wait for cooldown to expire
    time.sleep(2.1)
    alert_manager.send_alert("node_1", "Fall Detected", 0.92)
    assert mock_post.call_count == 2

    # Different alert type should bypass cooldown
    alert_manager.send_alert("node_1", "Zone Intrusion", 0.85)
    assert mock_post.call_count == 3
