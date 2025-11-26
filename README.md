# Federated Learning for Gastric Cancer Classification

A complete federated learning system for gastric cancer classification using 30,000 images across 8 classes, distributed among 3 simulated hospitals.

## 🏥 Dataset Classes
- **TUM**: Tumor
- **STR**: Stroma
- **NOR**: Normal
- **MUS**: Muscle
- **MUC**: Mucosa
- **LYM**: Lymphocyte
- **DEB**: Debris
- **ADI**: Adipose

## 📁 Project Structure
```
FedGastricCancer/
├── server/
│   ├── server.py               # FL server implementation
│   └── aggregator.py           # Custom aggregation strategies
├── client/
│   ├── client.py               # Flower client implementation
│   ├── train.py                # Local training logic
│   ├── dataset.py              # Data loading and augmentation
│   └── data/
│       ├── hospital_1/         # Hospital 1 data (10k images)
│       ├── hospital_2/         # Hospital 2 data (10k images)
│       └── hospital_3/         # Hospital 3 data (10k images)
├── models/
│   └── mobilenetv3.py         # MobileNetV3 model definition
├── dataset_full/              # Your 30k images go here
├── partition_dataset.py       # Dataset partitioning script
├── run_fl.sh                  # Linux/Mac run script
├── run_fl.bat                 # Windows run script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Create dataset structure
python partition_dataset.py create_structure
```

### 2. Add Your Dataset
Place your 30,000 gastric cancer images in the following structure:
```
dataset_full/
├── TUM/          # Tumor images
├── STR/          # Stroma images
├── NOR/          # Normal images
├── MUS/          # Muscle images
├── MUC/          # Mucosa images
├── LYM/          # Lymphocyte images
├── DEB/          # Debris images
└── ADI/          # Adipose images
```

### 3. Partition Dataset
```bash
python partition_dataset.py
```
This will split your 30k images across 3 hospitals with IID distribution.

### 4. Run Federated Learning

#### Option A: Complete System (Recommended)
```bash
# Linux/Mac
./run_fl.sh

# Windows
run_fl.bat
```

#### Option B: Manual Setup
```bash
# Terminal 1: Start server
python server/server.py --rounds 10 --min-clients 3

# Terminal 2: Start Hospital 1
python client/client.py 1

# Terminal 3: Start Hospital 2
python client/client.py 2

# Terminal 4: Start Hospital 3
python client/client.py 3
```

## 🔧 Configuration

### Server Configuration
```bash
python server/server.py \
    --strategy fedavg \
    --rounds 10 \
    --min-clients 3 \
    --local-epochs 2 \
    --learning-rate 1e-4 \
    --address 0.0.0.0:8080
```

### Client Configuration
```bash
python client/client.py 1 \
    --server 127.0.0.1:8080 \
    --model small
```

## 🧠 Model Architecture

### MobileNetV3-Small
- **Input**: 224x224 RGB images
- **Output**: 8 classes
- **Parameters**: ~2.9M
- **Pretrained**: ImageNet weights

### Data Augmentation
- **Training**: Random crop, flip, rotation, color jitter
- **Validation**: Center crop only
- **Normalization**: ImageNet statistics

## 📊 Aggregation Strategies

### 1. FedAvg (Default)
- Weighted average based on sample counts
- Standard federated averaging

### 2. FedAMP + FIM (Advanced)
- Adaptive model personalization
- Fisher Information Matrix weighting
- Enhanced privacy and performance

## 🎯 Training Process

1. **Initialization**: Server creates global model
2. **Local Training**: Each hospital trains on local data
3. **Aggregation**: Server combines model updates
4. **Evaluation**: Global model performance assessment
5. **Iteration**: Repeat for specified rounds

## 📈 Monitoring

The system provides real-time monitoring:
- Training/validation metrics per hospital
- Aggregation statistics
- Global model performance
- Client participation rates

## 🔍 Advanced Features

### Custom Aggregation
```python
from server.aggregator import FedAMPFIMAggregator

# Use advanced aggregation
strategy = FedAMPFIMAggregator(alpha=0.1, beta=0.01)
```

### Model Variants
```python
# Use different model architectures
python client/client.py 1 --model large  # MobileNetV3-Large
```

### Data Augmentation Levels
```python
# Adjust augmentation strength
# In client/dataset.py: augmentation_strength = "strong"
```

## 🐛 Troubleshooting

### Common Issues

1. **Dataset not found**
   ```bash
   python partition_dataset.py create_structure
   # Add your images to dataset_full/
   python partition_dataset.py
   ```

2. **Client connection failed**
   - Ensure server is running first
   - Check server address and port
   - Verify firewall settings

3. **CUDA out of memory**
   - Reduce batch size in client/train.py
   - Use CPU: set device to "cpu"

4. **Import errors**
   ```bash
   pip install -r requirements.txt
   ```

## 📝 Customization

### Adding New Hospitals
1. Modify `partition_dataset.py` to create more hospital directories
2. Update `run_fl.sh` to start additional clients
3. Adjust `min_clients` parameter in server configuration

### Custom Model Architecture
1. Modify `models/mobilenetv3.py`
2. Add new model definitions
3. Update client initialization

### Advanced Aggregation
1. Implement custom strategies in `server/aggregator.py`
2. Add new aggregation methods
3. Configure server to use new strategies

## 📚 Research Integration

This system is designed to support research on:
- Federated learning for medical imaging
- Privacy-preserving machine learning
- Multi-institutional collaboration
- Gastric cancer classification

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Implement changes
4. Test thoroughly
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Flower framework for federated learning
- PyTorch for deep learning
- Medical imaging research community
- Gastric cancer research datasets
