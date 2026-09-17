import numpy as np

def determine_model_tier(vram_mb, config):
    threshold = config['adaptive']['vram_threshold_mb']
    if vram_mb < threshold:
        return config['models']['nano']
    return config['models']['small']

def decompress_weights(sparse_updates, original_weights):
    # In our simulation, the client sends zeroed-out non-top-k values directly.
    # A true implementation would send indices and values to reconstruct.
    new_weights = []
    for update, orig in zip(sparse_updates, original_weights):
        new_weights.append(orig + update)
    return new_weights

def apply_fedprox_regularization(local_weights, global_weights, mu=0.01):
    """
    Apply FedProx proximal term (FLM-03) to stabilize aggregation across non-IID partitions.
    In standard FedProx, this is added to the local loss during training. If calculated server-side
    on weight deltas, we apply the proximal penalty directly to pull weights closer to global.
    """
    regularized = []
    for local_w, global_w in zip(local_weights, global_weights):
        proximal_term = mu * (local_w - global_w)
        regularized.append(local_w - proximal_term)
    return regularized

def knowledge_distillation_reconciliation(nano_params, small_params, temperature=4.0):
    """
    Simulated KD Reconciliation step (ARM-06).
    In a true KD scenario, a proxy dataset is required on the server to pass through
    both the Teacher (Small) and Student (Nano) to match logits with temperature scaling.
    Since raw data is strictly kept at the edge (NFR-S01), this simulates the gradient
    distillation by computing a proxy distillation mask based on L2 norms and interpolating.
    """
    reconciled_nano = []
    # Distill matching layers (early layers often match shape between Nano and Small)
    for n_param, s_param in zip(nano_params, small_params):
        if n_param.shape == s_param.shape:
            # Simulate logit distillation scaling factor
            norm_ratio = np.linalg.norm(s_param) / (np.linalg.norm(n_param) + 1e-8)
            kd_factor = np.exp(-norm_ratio / temperature)
            # Pull Nano closer to Small's representation
            distilled_param = n_param * (1 - kd_factor) + s_param * kd_factor
            reconciled_nano.append(distilled_param)
        else:
            reconciled_nano.append(n_param)
    return reconciled_nano
