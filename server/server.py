"""
Federated Learning Server for Gastric Cancer Classification
Robust global-model saving for different Flower versions and strategy implementations.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import flwr as fl
import torch
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.mobilenetv3 import get_model
from aggregator import get_strategy, get_on_fit_config_fn, get_on_evaluate_config_fn


class GastricCancerServer:
    """Federated Learning Server for Gastric Cancer Classification"""

    def __init__(
        self,
        strategy_name: str = "fedavg",
        num_rounds: int = 10,
        min_clients: int = 3,
        local_epochs: int = 2,
        learning_rate: float = 1e-4,
        server_address: str = "0.0.0.0:8080",
    ):
        self.strategy_name = strategy_name
        self.num_rounds = num_rounds
        self.min_clients = min_clients
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.server_address = server_address

        # Directory for saving global model and plots
        self.save_dir = Path("server/saved_models")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Create FL strategy
        # Avoid passing Flower args that may not be supported in older/newer versions
        self.strategy = get_strategy(
            strategy_name=strategy_name,
            fraction_fit=1.0,
            min_fit_clients=min_clients,
            min_available_clients=min_clients,
            on_fit_config_fn=get_on_fit_config_fn(local_epochs, learning_rate),
            on_evaluate_config_fn=get_on_evaluate_config_fn(),
            accept_failures=True,
        )

        print(f"✅ FL Server initialized")
        print(f"  Strategy: {strategy_name}")
        print(f"  Rounds: {num_rounds}")
        print(f"  Min clients required: {min_clients}")
        print(f"  Local Epochs: {local_epochs}")
        print(f"  Learning Rate: {learning_rate}")
        print(f"  Address: {server_address}")

    def start(self):
        """Start FL Server and save global model after training finishes."""
        print(f"\n🚀 Starting Federated Learning Server...")
        print(f"Waiting for {self.min_clients} clients to connect...")
        print("Press Ctrl+C to stop the server at any time.\n")

        try:
            # start_server returns a History-like object in Flower
            history = fl.server.start_server(
                server_address=self.server_address,
                config=fl.server.ServerConfig(num_rounds=self.num_rounds),
                strategy=self.strategy,
            )

            print("\n🏁 Training complete! Attempting to save global model...")
            self.save_global_model(history=history)

            # Try to print metrics summary if available
            self.print_final_metrics(history)

        except KeyboardInterrupt:
            print("\n🛑 Server stopped by user")
        except Exception as e:
            print(f"❌ Server Error: {e}")
            sys.exit(1)

    def print_final_metrics(self, history: Optional[Dict[str, Any]]):
        """Print final aggregated metrics if available in history."""
        # History returned by Flower may be a History object (dict-like) or None
        try:
            if history is None:
                print("⚠️ No history object returned by Flower server.")
                return

            # Try to access history as dict-like
            metrics_distributed = None
            if isinstance(history, dict):
                metrics_distributed = history.get("metrics_distributed", {})
            else:
                # Some Flower versions return a History object with dict attributes
                # Attempt commonly-used attributes
                metrics_distributed = getattr(history, "metrics_distributed", None) or getattr(history, "metrics", None)

            if not metrics_distributed:
                print("⚠️ No distributed metrics available in history.")
                return

            # Print last-round aggregated metrics if present
            rounds = sorted(metrics_distributed.keys())
            last_round = rounds[-1] if rounds else None
            if last_round is not None:
                last_metrics = metrics_distributed[last_round]
                print("\n" + "=" * 60)
                print(f"📊 FINAL AGGREGATED METRICS (Round {last_round})")
                print("=" * 60)
                # Generic keys used in this project (if aggregator added them)
                for key in [
                    "aggregated_train_loss",
                    "aggregated_train_accuracy",
                    "aggregated_train_precision",
                    "aggregated_train_recall",
                    "aggregated_train_f1",
                    "aggregated_val_loss",
                    "aggregated_val_accuracy",
                    "aggregated_val_precision",
                    "aggregated_val_recall",
                    "aggregated_val_f1",
                ]:
                    if key in last_metrics:
                        val = last_metrics[key]
                        # print nicely
                        print(f"  {key}: {val}")
                print("=" * 60 + "\n")
        except Exception as exc:
            print(f"⚠️ Failed to print final metrics from history: {exc}")

    def save_global_model(self, history: Optional[Dict[str, Any]] = None):
        """
        Save the aggregated global model.

        This function attempts multiple ways to locate the final aggregated parameters:
         - self.strategy.parameters (older Flower or some custom strategies)
         - self.strategy.strategy_state["parameters"] (common place for storing in Flower + custom aggregator)
         - history (if history contains 'parameters' or similar)
        """
        try:
            parameters = None

            # 1) Strategy may expose a `parameters` attribute (older custom strategies)
            parameters = getattr(self.strategy, "parameters", None)
            if parameters is not None:
                print("ℹ️ Found parameters at self.strategy.parameters")

            # 2) Strategy may have a strategy_state dict containing 'parameters'
            if parameters is None:
                strategy_state = getattr(self.strategy, "strategy_state", None)
                if strategy_state and isinstance(strategy_state, dict) and "parameters" in strategy_state:
                    parameters = strategy_state["parameters"]
                    print("ℹ️ Found parameters at self.strategy.strategy_state['parameters']")

            # 3) The aggregator might have stored final parameters as `self.parameters` attribute
            if parameters is None:
                p_attr = getattr(self.strategy, "parameters", None)
                if p_attr is not None:
                    parameters = p_attr
                    print("ℹ️ Found parameters at self.strategy.parameters (fallback)")

            # 4) Some implementations put parameters in history. Try to find them.
            if parameters is None and history is not None:
                try:
                    if isinstance(history, dict):
                        # Some wrappers may include 'parameters' key
                        if "parameters" in history:
                            parameters = history["parameters"]
                            print("ℹ️ Found parameters in history['parameters']")
                    else:
                        # History object might have an attribute for parameters
                        parameters = getattr(history, "parameters", None)
                        if parameters is not None:
                            print("ℹ️ Found parameters at history.parameters")
                except Exception:
                    parameters = None

            if parameters is None:
                print("⚠️ No global model parameters found (trained parameters not available).")
                print("Make sure your aggregator stores the aggregated parameters into strategy_state or strategy.parameters.")
                return

            # Convert federated Parameters to NDArrays (list of numpy arrays)
            try:
                ndarrays = fl.common.parameters_to_ndarrays(parameters)
            except Exception as e:
                # If conversion fails, try to assume parameters are already ndarrays
                print(f"ℹ️ parameters_to_ndarrays conversion failed with: {e}; attempting to treat 'parameters' as NDArrays directly.")
                ndarrays = parameters  # may raise later if wrong type

            # Build model and load weights
            model = get_model(num_classes=8, variant="small", pretrained=False)

            # Iterate through model parameters and ndarray list
            model_params = list(model.parameters())
            if len(model_params) != len(ndarrays):
                # It's possible that the shapes differ because some frameworks include optimizer state or different ordering.
                # Try to align by numeric count but warn the user.
                print("⚠️ Parameter count mismatch: model has", len(model_params), "tensors but aggregated parameters have", len(ndarrays))
                # Attempt zip load (safest approach)
            # Load tensors into model params
            for p, arr in zip(model_params, ndarrays):
                # arr may be numpy array or list; ensure torch tensor and correct dtype
                t = torch.tensor(arr, dtype=p.data.dtype)
                if t.shape != p.data.shape:
                    # attempt reshape only if total elements match
                    if t.numel() == p.data.numel():
                        t = t.view_as(p.data)
                        print(f"ℹ️ Reshaped parameter from {arr.shape} -> {p.data.shape}")
                    else:
                        print(f"⚠️ Shape mismatch for a parameter: model {p.data.shape} vs incoming {t.shape}; skipping this parameter.")
                        continue
                p.data.copy_(t)

            model_path = self.save_dir / "global_model.pth"
            torch.save(model.state_dict(), model_path)
            print(f"💾 Global model saved successfully at: {model_path}")

        except Exception as e:
            print(f"❌ Failed to save global model: {e}")

    # Optionally keep helper to plot accuracy (if you store metrics)
    def plot_accuracy_graph(self, metrics_by_round: Dict[int, Dict[str, float]]):
        """
        Optional helper: plot training/validation accuracy if you collected them in a dict:
        {round: {"train_acc": ..., "val_acc": ...}, ...}
        """
        if not metrics_by_round:
            print("⚠️ No metrics provided for plotting.")
            return

        rounds = sorted(metrics_by_round.keys())
        train_acc = [metrics_by_round[r].get("train_acc", None) for r in rounds]
        val_acc = [metrics_by_round[r].get("val_acc", None) for r in rounds]

        plt.figure(figsize=(8, 5))
        plt.plot(rounds, train_acc, marker="o", label="Train Acc")
        plt.plot(rounds, val_acc, marker="s", label="Val Acc")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plot_path = self.save_dir / "accuracy_plot.png"
        plt.savefig(plot_path, dpi=200)
        print(f"Saved accuracy plot at: {plot_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Start FL Server for Gastric Cancer Classification")
    parser.add_argument("--strategy", default="fedavg", choices=["fedavg", "fedamp_fim"])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--min-clients", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--address", default="0.0.0.0:8080")
    args = parser.parse_args()

    server = GastricCancerServer(
        strategy_name=args.strategy,
        num_rounds=args.rounds,
        min_clients=args.min_clients,
        local_epochs=args.local_epochs,
        learning_rate=args.learning_rate,
        server_address=args.address,
    )
    server.start()


if __name__ == "__main__":
    main()
