# Pneumonia X-Ray Classification System

A deep learning model that detects pneumonia from chest X-ray images using PyTorch and transfer learning with ResNet18.

## Sample X-ray Images

<!-- Insert sample images here -->
<p align="center">
  <img src="test/NORMAL/IM-0001-0001.jpeg" width="400" alt="Normal X-ray">
  <img src="images/pneumonia_sample.jpg" width="400" alt="Pneumonia X-ray">
</p>
<p align="center">
  <em>Left: Normal chest X-ray. Right: Chest X-ray showing pneumonia.</em>
</p>


## Project Overview

This project implements a medical imaging classification system that can identify pneumonia from chest X-rays. The system utilizes transfer learning with a pre-trained ResNet18 model, which is fine-tuned on a dataset of labeled chest X-ray images.

## Features

- Transfer learning with ResNet18 architecture
- Data preprocessing pipeline for X-ray images
- GPU acceleration with PyTorch
- Model evaluation with validation and test datasets
- Trained on chest X-ray images categorized as normal or pneumonia

## Dataset

The model is trained on the Chest X-Ray Images (Pneumonia) dataset which contains:
- Training, validation, and test sets
- Two categories: NORMAL and PNEUMONIA
- X-ray images in various formats (.jpg, .jpeg, .png, etc.)

The dataset should be organized in the following structure:
```
data/chest_xray/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/pneumonia-xray-classification.git
cd pneumonia-xray-classification
```

2. Install dependencies:
```
pip install -r requirements.txt
```

## Usage

### Training the model

Run the training script:
```
python model.py
```

This will:
- Load and preprocess the dataset
- Initialize a ResNet18 model with pre-trained weights
- Train the model for 10 epochs
- Validate and test the model's performance
- Save the trained model as `pneumonia_classifier.pth`

### Using the trained model for predictions

```python
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('pneumonia_classifier.pth', map_location=device))
model.to(device)
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and classify an image
image = Image.open('path_to_xray_image.jpg').convert('RGB')
image = transform(image).unsqueeze(0).to(device)
with torch.no_grad():
    output = model(image)
    _, pred = torch.max(output, 1)
    result = "PNEUMONIA" if pred.item() == 1 else "NORMAL"
print(f"Prediction: {result}")
```

## Model Performance

The model is evaluated using accuracy metrics on both validation and test datasets. Performance is monitored during training to prevent overfitting.

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- scikit-learn
- Pillow
- CUDA (optional, for GPU acceleration)

## Future Improvements

- Implement more advanced data augmentation techniques
- Explore other CNN architectures like DenseNet or EfficientNet
- Add visualization tools for model interpretability
- Develop a web interface for easy model usage

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Dataset from [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- Implementation uses PyTorch and torchvision libraries
