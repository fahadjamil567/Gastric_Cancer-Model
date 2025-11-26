"""
Dataset Loading and Data Augmentation for Gastric Cancer Classification
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, Optional, Dict, Any

class GastricCancerDataset(Dataset):
    """
    Custom dataset for gastric cancer classification
    """
    
    def __init__(self, 
                 data_dir: str,
                 transform: Optional[transforms.Compose] = None,
                 split: str = "train"):
        """
        Initialize dataset
        
        Args:
            data_dir: Directory containing class folders
            transform: Data augmentation transforms
            split: Dataset split ("train" or "val")
        """
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        
        # Define class mapping
        self.class_names = ["TUM", "STR", "NOR", "MUS", "MUC", "LYM", "DEB", "ADI"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        
        # Load dataset
        self.samples = self._load_dataset()
        
        print(f"Loaded {len(self.samples)} {split} samples from {data_dir}")
    
    def _load_dataset(self):
        """Load dataset samples"""
        samples = []
        
        for class_name in self.class_names:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory {class_dir} not found")
                continue
                
            class_idx = self.class_to_idx[class_name]
            
            # Get all image files in the class directory
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(class_dir, filename)
                    samples.append((img_path, class_idx))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image if loading fails
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_data_transforms(split: str = "train", 
                       image_size: int = 224,
                       augmentation_strength: str = "medium") -> transforms.Compose:
    """
    Get data augmentation transforms
    
    Args:
        split: Dataset split ("train" or "val")
        image_size: Target image size
        augmentation_strength: Augmentation strength ("light", "medium", "strong")
    
    Returns:
        Composed transforms
    """
    
    if split == "train":
        if augmentation_strength == "light":
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        elif augmentation_strength == "medium":
            transform = transforms.Compose([
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomCrop((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        elif augmentation_strength == "strong":
            transform = transforms.Compose([
                transforms.Resize((image_size + 64, image_size + 64)),
                transforms.RandomCrop((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(degrees=30),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            raise ValueError(f"Unknown augmentation strength: {augmentation_strength}")
    else:  # validation/test
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform

def get_data_loader(data_dir: str,
                   batch_size: int = 32,
                   split: str = "train",
                   num_workers: int = 4,
                   shuffle: bool = True,
                   image_size: int = 224,
                   augmentation_strength: str = "medium") -> DataLoader:
    """
    Create data loader for gastric cancer dataset
    
    Args:
        data_dir: Directory containing class folders
        batch_size: Batch size
        split: Dataset split ("train" or "val")
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        image_size: Target image size
        augmentation_strength: Augmentation strength
    
    Returns:
        DataLoader instance
    """
    
    # Get transforms
    transform = get_data_transforms(split, image_size, augmentation_strength)
    
    # Create dataset
    dataset = GastricCancerDataset(data_dir, transform, split)
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True if split == "train" else False
    )
    
    return dataloader

def get_class_weights(data_dir: str) -> torch.Tensor:
    """
    Calculate class weights for imbalanced dataset
    
    Args:
        data_dir: Directory containing class folders
    
    Returns:
        Tensor of class weights
    """
    class_counts = []
    class_names = ["TUM", "STR", "NOR", "MUS", "MUC", "LYM", "DEB", "ADI"]
    
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
            class_counts.append(count)
        else:
            class_counts.append(1)  # Avoid division by zero
    
    # Calculate weights (inverse frequency)
    total_samples = sum(class_counts)
    class_weights = [total_samples / (len(class_counts) * count) for count in class_counts]
    
    return torch.FloatTensor(class_weights)

def print_dataset_info(data_dir: str):
    """Print dataset information"""
    class_names = ["TUM", "STR", "NOR", "MUS", "MUC", "LYM", "DEB", "ADI"]
    
    print(f"Dataset information for {data_dir}:")
    print("-" * 50)
    
    total_samples = 0
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
            print(f"{class_name:>4}: {count:>6} samples")
            total_samples += count
        else:
            print(f"{class_name:>4}: {0:>6} samples (directory not found)")
    
    print("-" * 50)
    print(f"Total: {total_samples:>6} samples")

if __name__ == "__main__":
    # Test dataset loading
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
        print_dataset_info(data_dir)
        
        # Test data loader
        print("\nTesting data loader...")
        dataloader = get_data_loader(data_dir, batch_size=4, split="train")
        
        for i, (images, labels) in enumerate(dataloader):
            print(f"Batch {i}: images shape {images.shape}, labels shape {labels.shape}")
            if i >= 2:  # Test only first 3 batches
                break
    else:
        print("Usage: python dataset.py <data_directory>")
        print("Example: python dataset.py client/data/hospital_1/train")
