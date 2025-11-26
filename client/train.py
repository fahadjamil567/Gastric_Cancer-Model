"""
Training Logic for Federated Learning Client
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, Tuple, Optional
import time
import os
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from dataset import get_data_loader, get_class_weights
from models.mobilenetv3 import get_model

class ClientTrainer:
    """
    Client trainer for federated learning
    """
    
    def __init__(self, 
                 data_dir: str,
                 model_variant: str = "small",
                 device: Optional[torch.device] = None,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-4,
                 batch_size: int = 32,
                 num_workers: int = 4):
        """
        Initialize client trainer
        
        Args:
            data_dir: Directory containing training data
            model_variant: Model variant ("small" or "large")
            device: Device to use for training
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            batch_size: Batch size
            num_workers: Number of worker processes
        """
        self.data_dir = data_dir
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Initialize model
        self.model = get_model(num_classes=8, variant=model_variant).to(self.device)
        
        # Create data loaders
        self.train_loader = get_data_loader(
            os.path.join(data_dir, "train"),
            batch_size=batch_size,
            split="train",
            num_workers=num_workers,
            shuffle=True
        )
        
        self.val_loader = get_data_loader(
            os.path.join(data_dir, "val"),
            batch_size=batch_size,
            split="val",
            num_workers=num_workers,
            shuffle=False
        )
        
        # Get class weights for imbalanced dataset
        self.class_weights = get_class_weights(os.path.join(data_dir, "train")).to(self.device)
        
        # Initialize loss and optimizer
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
        
        print(f"Client trainer initialized on {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Store predictions and targets for metrics calculation
            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy())
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {100. * correct / total:.2f}%')
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        # Calculate precision, recall, and F1 score
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Use macro averaging for multi-class classification
        precision = precision_score(all_targets, all_preds, average='macro', zero_division=0) * 100
        recall = recall_score(all_targets, all_preds, average='macro', zero_division=0) * 100
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0) * 100
        
        return {
            'train_loss': avg_loss,
            'train_accuracy': accuracy,
            'train_precision': precision,
            'train_recall': recall,
            'train_f1': f1
        }
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model
        
        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # Store predictions and targets for metrics calculation
                all_preds.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        # Calculate precision, recall, and F1 score
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Use macro averaging for multi-class classification
        precision = precision_score(all_targets, all_preds, average='macro', zero_division=0) * 100
        recall = recall_score(all_targets, all_preds, average='macro', zero_division=0) * 100
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0) * 100
        
        return {
            'val_loss': avg_loss,
            'val_accuracy': accuracy,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1
        }
    
    def train(self, epochs: int = 2, save_best: bool = False) -> Dict[str, Any]:
        """
        Train the model for specified epochs
        
        Args:
            epochs: Number of epochs to train
            save_best: Whether to save the best model
        
        Returns:
            Dictionary containing training results
        """
        print(f"Starting training for {epochs} epochs...")
        
        best_val_acc = 0.0
        train_history = []
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_metrics = self.train_epoch(epoch + 1)
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_metrics['val_loss'])
            
            # Print metrics
            print(f"Train Loss: {train_metrics['train_loss']:.4f}, "
                  f"Train Acc: {train_metrics['train_accuracy']:.2f}%, "
                  f"Precision: {train_metrics['train_precision']:.2f}%, "
                  f"Recall: {train_metrics['train_recall']:.2f}%, "
                  f"F1: {train_metrics['train_f1']:.2f}%")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}, "
                  f"Val Acc: {val_metrics['val_accuracy']:.2f}%, "
                  f"Precision: {val_metrics['val_precision']:.2f}%, "
                  f"Recall: {val_metrics['val_recall']:.2f}%, "
                  f"F1: {val_metrics['val_f1']:.2f}%")
            
            # Save best model
            if save_best and val_metrics['val_accuracy'] > best_val_acc:
                best_val_acc = val_metrics['val_accuracy']
                self.save_model("best_model.pth")
                print(f"New best model saved with val accuracy: {best_val_acc:.2f}%")
            
            # Store history
            epoch_metrics = {**train_metrics, **val_metrics}
            train_history.append(epoch_metrics)
        
        return {
            'train_history': train_history,
            'best_val_accuracy': best_val_acc,
            'final_metrics': train_history[-1] if train_history else {}
        }
    
    def get_parameters(self) -> list:
        """Get model parameters as numpy arrays"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: list):
        """Set model parameters from numpy arrays"""
        params_dict = dict(zip(self.model.state_dict().keys(), parameters))
        state_dict = {k: torch.tensor(v) for k, v in params_dict.items()}
        self.model.load_state_dict(state_dict)
    
    def save_model(self, filename: str):
        """Save model to file"""
        model_path = Path(self.data_dir).parent / filename
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, filename: str):
        """Load model from file"""
        model_path = Path(self.data_dir).parent / filename
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Model loaded from {model_path}")
        else:
            print(f"Model file {model_path} not found")

def train_client(data_dir: str, 
                epochs: int = 2,
                model_variant: str = "small",
                **kwargs) -> Dict[str, Any]:
    """
    Train a client model
    
    Args:
        data_dir: Directory containing training data
        epochs: Number of epochs to train
        model_variant: Model variant
        **kwargs: Additional arguments for ClientTrainer
    
    Returns:
        Training results
    """
    trainer = ClientTrainer(data_dir, model_variant=model_variant, **kwargs)
    results = trainer.train(epochs=epochs)
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python train.py <data_directory> [epochs] [model_variant]")
        print("Example: python train.py client/data/hospital_1 2 small")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    model_variant = sys.argv[3] if len(sys.argv) > 3 else "small"
    
    print(f"Training client with data from {data_dir}")
    print(f"Epochs: {epochs}, Model variant: {model_variant}")
    
    results = train_client(data_dir, epochs=epochs, model_variant=model_variant)
    
    print("\nTraining completed!")
    print(f"Best validation accuracy: {results['best_val_accuracy']:.2f}%")
    print(f"Final metrics: {results['final_metrics']}")
