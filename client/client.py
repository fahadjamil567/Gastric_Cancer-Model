"""
Flower Client Implementation for Gastric Cancer Federated Learning
"""

import flwr as fl
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.mobilenetv3 import get_model
from train import ClientTrainer


class GastricCancerClient(fl.client.NumPyClient):
    """
    Flower client for gastric cancer classification
    """

    def __init__(self,
                 hospital_id: str,
                 data_dir: str,
                 model_variant: str = "small",
                 device: Optional[torch.device] = None):
        """
        Initialize the client
        """
        self.hospital_id = hospital_id
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize trainer
        self.trainer = ClientTrainer(
            data_dir=data_dir,
            model_variant=model_variant,
            device=self.device
        )

        print(f"\n✅ Client initialized for: {hospital_id}")
        print(f"Device: {self.device}")
        print(f"Data directory: {data_dir}")
        print(f"Training samples: {len(self.trainer.train_loader.dataset)}")
        print(f"Validation samples: {len(self.trainer.val_loader.dataset)}")

    # ------------------ FLWR Methods ------------------

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        """Return model parameters"""
        return self.trainer.get_parameters()

    def fit(self,
            parameters: List[np.ndarray],
            config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict[str, str]]:
        """Train the model locally"""
        print(f"\n🏥 {self.hospital_id}: Starting local training...")

        # Load global parameters
        self.trainer.set_parameters(parameters)

        # Extract training config
        epochs = int(config.get("local_epochs", 2))
        lr = float(config.get("learning_rate", 1e-4))

        for group in self.trainer.optimizer.param_groups:
            group["lr"] = lr

        results = self.trainer.train(epochs=epochs, save_best=False)

        updated_params = self.trainer.get_parameters()
        num_examples = len(self.trainer.train_loader.dataset)

        metrics = {
            "train_loss": results['final_metrics'].get('train_loss', 0.0),
            "train_accuracy": results['final_metrics'].get('train_accuracy', 0.0),
            "train_precision": results['final_metrics'].get('train_precision', 0.0),
            "train_recall": results['final_metrics'].get('train_recall', 0.0),
            "train_f1": results['final_metrics'].get('train_f1', 0.0),
            "val_loss": results['final_metrics'].get('val_loss', 0.0),
            "val_accuracy": results['final_metrics'].get('val_accuracy', 0.0),
            "val_precision": results['final_metrics'].get('val_precision', 0.0),
            "val_recall": results['final_metrics'].get('val_recall', 0.0),
            "val_f1": results['final_metrics'].get('val_f1', 0.0),
            "hospital_id": self.hospital_id
        }

        print(f"\n🏥 {self.hospital_id}: Training Complete")
        print(f"  Train Loss: {metrics['train_loss']:.4f} | Train Acc: {metrics['train_accuracy']:.2f}% | "
              f"Precision: {metrics['train_precision']:.2f}% | Recall: {metrics['train_recall']:.2f}% | "
              f"F1: {metrics['train_f1']:.2f}%")
        print(f"  Val Loss:   {metrics['val_loss']:.4f} | Val Acc:   {metrics['val_accuracy']:.2f}% | "
              f"Precision: {metrics['val_precision']:.2f}% | Recall: {metrics['val_recall']:.2f}% | "
              f"F1: {metrics['val_f1']:.2f}%")

        # ✅ Save local model after training
        os.makedirs("client/saved_models", exist_ok=True)
        model_path = f"client/saved_models/{self.hospital_id}_model.pth"
        torch.save(self.trainer.model.state_dict(), model_path)
        print(f"💾 Local model saved at: {model_path}")

        return updated_params, num_examples, metrics

    def evaluate(self,
                 parameters: List[np.ndarray],
                 config: Dict[str, str]) -> Tuple[float, int, Dict[str, str]]:
        """Evaluate the model"""
        print(f"\n🏥 {self.hospital_id}: Starting evaluation...")
        self.trainer.set_parameters(parameters)

        val_metrics = self.trainer.validate()

        metrics = {
            "val_loss": val_metrics["val_loss"],
            "val_accuracy": val_metrics["val_accuracy"],
            "val_precision": val_metrics.get("val_precision", 0.0),
            "val_recall": val_metrics.get("val_recall", 0.0),
            "val_f1": val_metrics.get("val_f1", 0.0),
            "hospital_id": self.hospital_id
        }

        print(f"🏥 {self.hospital_id}: Evaluation Complete")
        print(f"  Val Loss: {metrics['val_loss']:.4f} | Val Accuracy: {metrics['val_accuracy']:.2f}% | "
              f"Precision: {metrics['val_precision']:.2f}% | Recall: {metrics['val_recall']:.2f}% | "
              f"F1: {metrics['val_f1']:.2f}%")

        return val_metrics["val_loss"], len(self.trainer.val_loader.dataset), metrics


# ------------------ CLIENT STARTUP ------------------

def start_client(hospital_id: str,
                 server_address: str = "127.0.0.1:8080",
                 data_dir: Optional[str] = None,
                 model_variant: str = "small"):
    """Start a Flower client for a given hospital."""
    if hospital_id.isdigit():
        hospital_name = f"hospital_{hospital_id}"
    else:
        hospital_name = hospital_id

    if data_dir is None:
        data_dir = f"client/data/{hospital_name}"

    if not os.path.exists(data_dir):
        print(f"❌ Error: Data directory not found: {data_dir}")
        print("Please make sure dataset is partitioned correctly.")
        return

    client = GastricCancerClient(
        hospital_id=hospital_name,
        data_dir=data_dir,
        model_variant=model_variant
    )

    print(f"\n🚀 Starting client: {hospital_name}")
    print(f"🔗 Connecting to server at {server_address}\n")

    try:
        fl.client.start_numpy_client(server_address=server_address, client=client)
    except Exception as e:
        print(f"❌ Error connecting to FL server: {e}")
        print("Ensure the server is running before starting clients.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start a Federated Learning client for Gastric Cancer Project")
    parser.add_argument("hospital_id", type=str, help="Hospital ID (1, 2, 3 or hospital_1, hospital_2...)")
    parser.add_argument("--server", default="127.0.0.1:8080", help="Federated Learning server address")
    parser.add_argument("--data_dir", help="Custom path to hospital data")
    parser.add_argument("--model", default="small", choices=["small", "large"], help="Model variant to use")

    args = parser.parse_args()

    start_client(
        hospital_id=args.hospital_id,
        server_address=args.server,
        data_dir=args.data_dir,
        model_variant=args.model
    )
