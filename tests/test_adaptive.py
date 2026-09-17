import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../server')))

from adaptive import determine_model_tier

def test_tier_assignment():
    # Test ARM-02
    config = {
        'adaptive': {'vram_threshold_mb': 4096},
        'models': {'nano': 'yolov8n.pt', 'small': 'yolov8s.pt'}
    }
    
    # Below threshold -> Nano
    tier = determine_model_tier(2048, config)
    assert tier == 'yolov8n.pt'
    
    # Above threshold -> Small
    tier = determine_model_tier(6144, config)
    assert tier == 'yolov8s.pt'

def test_top_k_sparsification():
    # Test ARM-05
    k = 0.1 # 10%
    
    # Create mock weight tensor of 100 elements
    np.random.seed(42)
    w = np.random.rand(10, 10)
    w_flat = w.flatten()
    
    num_top_k = max(1, int(len(w_flat) * k))
    indices = np.argpartition(np.abs(w_flat), -num_top_k)[-num_top_k:]
    mask = np.zeros_like(w_flat, dtype=bool)
    mask[indices] = True
    w_sparse = np.where(mask, w_flat, 0).reshape(w.shape)
    
    # Assert exactly 10 elements are non-zero
    non_zero_count = np.count_nonzero(w_sparse)
    assert non_zero_count == 10
    
    # Assert the non-zero elements are indeed the largest ones
    expected_threshold = np.sort(np.abs(w_flat))[-10]
    assert np.all(np.abs(w_sparse[w_sparse != 0]) >= expected_threshold)
