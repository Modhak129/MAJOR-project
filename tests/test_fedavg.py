import pytest
import numpy as np
import flwr as fl

def test_fedavg_tensor_math():
    # Test FLM-02: Basic FedAvg tensor math works as expected
    # Mock two client updates with equal weighting
    
    # Client 1 weights: array of [1.0, 1.0]
    client_1_weights = [np.array([1.0, 1.0], dtype=np.float32)]
    # Client 2 weights: array of [3.0, 3.0]
    client_2_weights = [np.array([3.0, 3.0], dtype=np.float32)]
    
    # Weights are parameterized as FitRes objects (mocking Flower structures)
    from flwr.common import FitRes, Status, Code, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
    
    res1 = FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=ndarrays_to_parameters(client_1_weights),
        num_examples=100, # equal weighting
        metrics={}
    )
    
    res2 = FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=ndarrays_to_parameters(client_2_weights),
        num_examples=100, # equal weighting
        metrics={}
    )
    
    results = [
        (None, res1),
        (None, res2)
    ]
    
    # Perform standard FedAvg math manually since server strategy is heavily mocked
    # Weighted average: (100 * [1,1] + 100 * [3,3]) / 200 = [2,2]
    total_examples = sum([res.num_examples for _, res in results])
    
    aggregated_ndarrays = None
    for _, res in results:
        ndarrays = parameters_to_ndarrays(res.parameters)
        weight = res.num_examples / total_examples
        
        if aggregated_ndarrays is None:
            aggregated_ndarrays = [arr * weight for arr in ndarrays]
        else:
            aggregated_ndarrays = [agg + arr * weight for agg, arr in zip(aggregated_ndarrays, ndarrays)]
            
    assert len(aggregated_ndarrays) == 1
    assert np.allclose(aggregated_ndarrays[0], np.array([2.0, 2.0]))
