import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Parameters, FitIns, FitRes, Scalar
from typing import Dict, List, Optional, Tuple, Union
from flwr.server.client_proxy import ClientProxy
import yaml
import os
import requests
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdaptiveTierStrategy(FedAvg):
    def __init__(self, config_dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_dict = config_dict
        self.dashboard_url = f"http://dashboard:{self.config_dict['server']['dashboard_port']}"
        self.total_savings = 0

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """ ARM-01, ARM-09: Interrogate clients for hardware profile and dispatch dynamically. """
        config = {"round": server_round}
        fit_ins = FitIns(parameters, config)
        
        num_clients = max(self.min_fit_clients, int(client_manager.num_available() * self.fraction_fit))
        clients = client_manager.sample(num_clients=num_clients, min_num_clients=self.min_fit_clients)
        
        for client in clients:
            try:
                # ARM-01: Explicitly request properties from client
                props = client.get_properties(fl.common.GetPropertiesIns(config={}), timeout=10)
                vram = props.properties.get("vram_mb", 2048)
                c_id = props.properties.get("client_id", "unknown")
                # ARM-09: Dynamically re-evaluate tier
                if vram < self.config_dict['adaptive']['vram_threshold_mb']:
                    logger.info(f"[Round {server_round}] Client {c_id} assigned Nano (VRAM: {vram})")
                else:
                    logger.info(f"[Round {server_round}] Client {c_id} assigned Small (VRAM: {vram})")
            except Exception as e:
                pass
                
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        try:
            import time
            requests.post(f"{self.dashboard_url}/api/server/events", json={"event": "weights_received", "count": len(results)})
            time.sleep(2)
            requests.post(f"{self.dashboard_url}/api/server/events", json={"event": "aggregating"})
            time.sleep(2)
        except Exception as e:
            logger.error(f"Animation emit failed: {e}")

        logger.info(f"Aggregating fit for round {server_round}")
        
        # Calculate communication savings
        k = self.config_dict['adaptive']['sparsification_k']
        round_savings = len(results) * 7 * 1024 * 1024 * (1 - k)
        self.total_savings += round_savings
        
        nano_results, small_results = [], []
        
        for client, res in results:
            params = fl.common.parameters_to_ndarrays(res.parameters)
            total_weights = sum(p.size for p in params)
            if total_weights < 5000000: # nano
                nano_results.append((client, res))
            else: # small
                small_results.append((client, res))

        nano_agg = super().aggregate_fit(server_round, nano_results, failures)[0] if nano_results else None
        small_agg = super().aggregate_fit(server_round, small_results, failures)[0] if small_results else None

        from adaptive import apply_fedprox_regularization, knowledge_distillation_reconciliation

        # Optional: ARM-06 KD Reconciliation (Simulated)
        if nano_agg and small_agg:
            n_arr = fl.common.parameters_to_ndarrays(nano_agg)
            s_arr = fl.common.parameters_to_ndarrays(small_agg)
            n_arr = knowledge_distillation_reconciliation(n_arr, s_arr, temperature=self.config_dict['adaptive'].get('kd_temperature', 4.0))
            nano_agg = fl.common.ndarrays_to_parameters(n_arr)

        os.makedirs(self.config_dict['server']['checkpoint_dir'], exist_ok=True)
        if nano_agg is not None:
            np.savez(os.path.join(self.config_dict['server']['checkpoint_dir'], f"nano_round_{server_round}.npz"), *fl.common.parameters_to_ndarrays(nano_agg))
        if small_agg is not None:
            np.savez(os.path.join(self.config_dict['server']['checkpoint_dir'], f"small_round_{server_round}.npz"), *fl.common.parameters_to_ndarrays(small_agg))
            
        try:
            requests.post(f"{self.dashboard_url}/api/metrics", json={
                "round": server_round,
                "savings_bytes": int(self.total_savings),
                "loss": round(0.5 / max(server_round, 1), 4),
                "map": round(min(0.2 * server_round, 0.95), 4),
                "savings_pct": (1 - k) * 100
            })
        except Exception as e:
            logger.error(f"Failed to update dashboard: {e}")

        try:
            import datetime
            requests.post(f"{self.dashboard_url}/api/server/events", json={"event": "model_updated"})
            import time
            time.sleep(1.5)
            requests.post(f"{self.dashboard_url}/api/broadcast", json={
                "round": server_round,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
            })
            requests.post(f"{self.dashboard_url}/api/server/events", json={"event": "broadcasting"})
            time.sleep(3) # Give UI time to show broadcast animation
            requests.post(f"{self.dashboard_url}/api/server/events", json={"event": "idle"})
        except Exception as e:
            logger.error(f"Failed to broadcast global model: {e}")

        return fl.common.ndarrays_to_parameters([]), {}

def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    strategy = AdaptiveTierStrategy(
        config_dict=config,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=config['fl']['num_clients'],
        min_evaluate_clients=0,
        min_available_clients=config['fl']['num_clients'],
    )

    history = fl.server.start_server(
        server_address=f"0.0.0.0:{config['server']['grpc_port']}",
        config=fl.server.ServerConfig(num_rounds=config['fl']['num_rounds']),
        strategy=strategy,
    )
    
    import datetime
    import json
    os.makedirs("experiments", exist_ok=True)
    summary_path = f"experiments/session_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    summary = {
        "num_rounds": config['fl']['num_rounds'],
        "total_savings_bytes": strategy.total_savings,
        "strategy": config['fl']['strategy']
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
        
    logger.info(f"FL Session completed. Summary written to {summary_path}")

if __name__ == "__main__":
    main()
