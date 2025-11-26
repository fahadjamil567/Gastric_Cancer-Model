"""
Dataset Partitioning Script for Federated Learning
Splits 30,000 gastric cancer images across 3 hospitals
"""

import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

def partition_dataset(source_dir="dataset_full", output_dir="client/data", num_hospitals=3):
    """
    Partition dataset into 3 hospitals with IID split
    
    Args:
        source_dir: Directory containing the full dataset with class folders
        output_dir: Output directory for hospital data
        num_hospitals: Number of hospitals to partition data into
    """
    
    # Define the 8 classes
    classes = ["TUM", "STR", "NOR", "MUS", "MUC", "LYM", "DEB", "ADI"]
    
    # Create hospital directories
    for h in range(1, num_hospitals + 1):
        hospital_dir = Path(output_dir) / f"hospital_{h}"
        hospital_dir.mkdir(parents=True, exist_ok=True)
        (hospital_dir / "train").mkdir(exist_ok=True)
        (hospital_dir / "val").mkdir(exist_ok=True)
        
        # Create class directories for each hospital
        for cls in classes:
            (hospital_dir / "train" / cls).mkdir(exist_ok=True)
            (hospital_dir / "val" / cls).mkdir(exist_ok=True)
    
    print(f"Partitioning dataset from {source_dir} to {output_dir}")
    print(f"Creating {num_hospitals} hospitals with IID split")
    
    # Process each class
    for cls in classes:
        class_dir = Path(source_dir) / cls
        
        if not class_dir.exists():
            print(f"Warning: Class directory {class_dir} not found. Skipping...")
            continue
            
        # Get all images in this class
        images = [f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        if not images:
            print(f"Warning: No images found in {class_dir}")
            continue
            
        print(f"Processing {len(images)} images for class {cls}")
        
        # Shuffle images randomly
        random.shuffle(images)
        
        # Split into roughly equal parts for each hospital
        chunk_size = len(images) // num_hospitals
        remainder = len(images) % num_hospitals
        
        start_idx = 0
        for h in range(1, num_hospitals + 1):
            # Calculate chunk size for this hospital
            current_chunk_size = chunk_size + (1 if h <= remainder else 0)
            end_idx = start_idx + current_chunk_size
            
            # Get images for this hospital
            hospital_images = images[start_idx:end_idx]
            
            # Split into train/val (80/20)
            if len(hospital_images) > 1:
                train_images, val_images = train_test_split(
                    hospital_images, 
                    test_size=0.2, 
                    random_state=42
                )
            else:
                train_images = hospital_images
                val_images = []
            
            # Copy training images
            for img_path in train_images:
                dest_path = Path(output_dir) / f"hospital_{h}" / "train" / cls / img_path.name
                shutil.copy2(img_path, dest_path)
            
            # Copy validation images
            for img_path in val_images:
                dest_path = Path(output_dir) / f"hospital_{h}" / "val" / cls / img_path.name
                shutil.copy2(img_path, dest_path)
            
            print(f"Hospital {h}: {len(train_images)} train, {len(val_images)} val images for class {cls}")
            start_idx = end_idx
    
    print("Dataset partitioning completed!")
    
    # Print summary
    for h in range(1, num_hospitals + 1):
        hospital_dir = Path(output_dir) / f"hospital_{h}"
        train_count = sum(len(list((hospital_dir / "train" / cls).glob("*"))) for cls in classes)
        val_count = sum(len(list((hospital_dir / "val" / cls).glob("*"))) for cls in classes)
        print(f"Hospital {h}: {train_count} train images, {val_count} val images")

def create_sample_structure():
    """Create sample directory structure for testing"""
    source_dir = Path("dataset_full")
    source_dir.mkdir(exist_ok=True)
    
    classes = ["TUM", "STR", "NOR", "MUS", "MUC", "LYM", "DEB", "ADI"]
    
    for cls in classes:
        class_dir = source_dir / cls
        class_dir.mkdir(exist_ok=True)
        print(f"Created directory: {class_dir}")
        print(f"Please add your {cls} class images to: {class_dir.absolute()}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "create_structure":
        create_sample_structure()
    else:
        # Check if source dataset exists
        if not Path("dataset_full").exists():
            print("Source dataset directory 'dataset_full' not found!")
            print("Please create the directory structure first by running:")
            print("python partition_dataset.py create_structure")
            print("\nThen add your 30,000 images organized by class folders:")
            print("dataset_full/")
            print("├── TUM/")
            print("├── STR/")
            print("├── NOR/")
            print("├── MUS/")
            print("├── MUC/")
            print("├── LYM/")
            print("├── DEB/")
            print("└── ADI/")
        else:
            partition_dataset()
