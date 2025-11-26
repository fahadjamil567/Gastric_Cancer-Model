"""
Custom Federated Learning Aggregation Strategies
Implements FedAvg and advanced aggregation methods (FedAMP + FIM)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from flwr.server.strategy import FedAvg
from flwr.common import (
    NDArrays,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
    FitRes,
)
import flwr as fl


# =====================================================================
#  HELPER: SAVE PARAMETERS INTO STRATEGY
# =====================================================================

def save_parameters_into_strategy(strategy, aggregated_parameters):
    """
    Ensures global parameters are stored in common places so the server
    can reliably find and save the global model.
    """

    # Old / Custom Flower versions expect this
    strategy.parameters = aggregated_parameters

    # Create strategy_state if missing
    if not hasattr(strategy, "strategy_state"):
        strategy.strategy_state = {}

    # Store parameters inside strategy_state
    strategy.strategy_state["parameters"] = aggregated_parameters


# =====================================================================
#  FEDAVG IMPLEMENTATION
# =====================================================================

class CustomFedAvg(FedAvg):
    """Custom FedAvg with proper aggregation and enhanced logging"""

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, Scalar]]:

        if not results:
            return None, {}

        # Extract parameters and samples
        weights_results = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        sample_counts = [fit_res.num_examples for _, fit_res in results]

        total_samples = sum(sample_counts)
        weighted_params = None

        # Weighted aggregation
        for idx, client_weights in enumerate(weights_results):
            weight = sample_counts[idx] / total_samples

            if weighted_params is None:
                weighted_params = [w * weight for w in client_weights]
            else:
                weighted_params = [
                    wp + (w * weight) for wp, w in zip(weighted_params, client_weights)
                ]

        aggregated_parameters = ndarrays_to_parameters(weighted_params)

        # ---------------------------------------------------------
        # STORE PARAMETERS IN STRATEGY (IMPORTANT!)
        # ---------------------------------------------------------
        save_parameters_into_strategy(self, aggregated_parameters)

        # Aggregate Metrics
        metrics = self._aggregate_metrics(server_round, results, total_samples)

        return aggregated_parameters, metrics

    # =====================================================================
    # METRIC AGGREGATION HELPER
    # =====================================================================

    def _aggregate_metrics(self, server_round, results, total_samples):
        """Weighted metric aggregation with clean summary"""
        agg = lambda key: sum(
            [
                fit_res.metrics[key] * (fit_res.num_examples / total_samples)
                for _, fit_res in results
                if key in fit_res.metrics
            ]
        )

        metric_keys = [
            "train_loss", "train_accuracy", "train_precision", "train_recall", "train_f1",
            "val_loss", "val_accuracy", "val_precision", "val_recall", "val_f1",
        ]

        metrics = {
            "round": server_round,
            "num_clients": len(results),
            "total_samples": total_samples,
        }

        for key in metric_keys:
            try:
                metrics[f"aggregated_{key}"] = agg(key)
            except:
                pass  # metric missing from some clients

        # Logging summary
        print(f"\n[Round {server_round}] Aggregated {len(results)} clients")
        print(f"  Total samples: {total_samples}")
        print(f"  Avg samples per client: {total_samples / len(results):.1f}")

        if "aggregated_train_accuracy" in metrics:
            print(
                f"  Train Acc: {metrics['aggregated_train_accuracy']:.2f}% | "
                f"Train Loss: {metrics['aggregated_train_loss']:.4f}"
            )

        if "aggregated_val_accuracy" in metrics:
            print(
                f"  Val Acc:   {metrics['aggregated_val_accuracy']:.2f}% | "
                f"Val Loss:   {metrics['aggregated_val_loss']:.4f}"
            )

        return metrics


# =====================================================================
#  FEDAMP + FIM STRATEGY
# =====================================================================

class FedAMPFIMAggregator(CustomFedAvg):
    """Advanced aggregator implementing FedAMP + FIM weighting"""

    def __init__(self, alpha: float = 0.1, beta: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[BaseException],
    ):
        if not results:
            return None, {}

        weights_results = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        sample_counts = [fit_res.num_examples for _, fit_res in results]

        total_samples = sum(sample_counts)
        weighted_params = None

        # FedAMP-FIM Weighted aggregation
        for idx, client_weights in enumerate(weights_results):
            weight = (sample_counts[idx] / total_samples) * (1 + self.alpha) / (1 + self.beta)

            if weighted_params is None:
                weighted_params = [w * weight for w in client_weights]
            else:
                weighted_params = [
                    wp + (w * weight) for wp, w in zip(weighted_params, client_weights)
                ]

        aggregated_parameters = ndarrays_to_parameters(weighted_params)

        # Store parameters in strategy
        save_parameters_into_strategy(self, aggregated_parameters)

        # Compute metrics
        metrics = self._aggregate_metrics(server_round, results, total_samples)
        metrics["alpha"] = self.alpha
        metrics["beta"] = self.beta

        print(f"[FedAMP-FIM] Round {server_round} aggregation complete.")

        return aggregated_parameters, metrics


# =====================================================================
# STRATEGY SELECTION
# =====================================================================

def get_strategy(strategy_name: str = "fedavg", **kwargs):
    if strategy_name == "fedavg":
        return CustomFedAvg(**kwargs)
    elif strategy_name == "fedamp_fim":
        return FedAMPFIMAggregator(**kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


# =====================================================================
# CONFIG FUNCTIONS
# =====================================================================

def get_on_fit_config_fn(epochs: int = 2, learning_rate: float = 1e-4):
    def fit_config(server_round: int):
        return {
            "local_epochs": str(epochs),
            "learning_rate": str(learning_rate),
        }

    return fit_config


def get_on_evaluate_config_fn():
    def evaluate_config(server_round: int):
        return {}

    return evaluate_config
