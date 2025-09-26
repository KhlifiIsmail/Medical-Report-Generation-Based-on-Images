# 🩻 Medical X-ray Classification System

**A Flask-based web application for automated chest X-ray analysis using Vision Transformer (ViT) models**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.0+-orange.svg)

## 🎯 Project Overview

This Flask application provides automated chest X-ray classification to assist in medical diagnosis. The system analyzes uploaded X-ray images and provides predictions for common chest conditions, particularly pneumonia detection.

**⚠️ MEDICAL DISCLAIMER**: This is an AI-powered tool for educational and research purposes only. Results should NOT be used for actual medical diagnosis without professional medical consultation.

## 🏗️ System Architecture

```
User Upload → Flask Server → Image Processing → AI Model → Medical Analysis → Results Display
```

### Core Components:

- **Flask Web Server**: Handles HTTP requests, file uploads, and response delivery
- **Vision Transformer (ViT)**: Deep learning model for medical image classification
- **Image Processor**: Preprocesses X-ray images for model compatibility
- **Classification Engine**: Provides medical predictions with confidence scores

## 🧠 AI Models & Classification Function

<img width="1193" height="717" alt="image" src="https://github.com/user-attachments/assets/bfa0f847-c8f4-43d5-912f-859585881467" />


### Model Architecture: Vision Transformer (ViT)

The system uses a **Vision Transformer** architecture, specifically designed for medical image analysis:

```python
class ViTForImageClassification(nn.Module):
    def __init__(self, num_classes=2):
        super(ViTForImageClassification, self).__init__()
        # Pre-trained ViT backbone from Google
        self.vit = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k")

        # Medical classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, 256),  # ViT hidden size: 768
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)  # Binary: Normal vs Pneumonia
        )
```

### Why Vision Transformer?

- **Attention Mechanism**: Focuses on relevant image regions like traditional radiologist examination
- **Global Context**: Captures long-range dependencies in chest X-rays
- **Medical Accuracy**: Superior performance on medical imaging tasks
- **Transfer Learning**: Leverages pre-trained weights from large-scale image datasets

### Classification Process

1. **Image Preprocessing**:

   ```python
   # Convert to RGB, resize to 224x224
   # Normalize with mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
   # Create tensor with batch dimension
   ```

2. **Model Inference**:

   ```python
   with torch.no_grad():
       logits = model(pixel_values)
       probabilities = torch.softmax(logits, dim=1)
       predicted_class = torch.argmax(probabilities, dim=1)
   ```

3. **Medical Interpretation**:
   ```python
   class_names = ['Normal', 'Pneumonia']
   confidence = probabilities.max().item() * 100
   ```

### Model Options

The system supports multiple model backends:

1. **Custom Trained Model**:

   - Files: `xray_classifier_huggingface.pth`, `processor.pkl`
   - Architecture: ViT with medical-specific fine-tuning

2. **Pre-trained Medical Models**:
   - `lxyuan/vit-xray-pneumonia-classification`
   - `nickmccullum/chest-xray-pneumonia-binary-classification`
   - `keremberke/chest-xray-classification`
   - Automatic fallback if primary model fails

## 📁 Project Structure

```
Medical-Report-Generation-Based-on-Images/
├── app.py                          # Flask web server
├── classify_function.py            # AI classification logic
├── classify_function_pretrained.py # Pre-trained model version
├── debug_model.py                  # Model debugging utilities
├── test_model_variability.py       # Model validation tests
├── requirements.txt                # Python dependencies
├── templates/
│   └── index.html                 # Web interface
├── uploads/                       # Temporary image storage
├── xray_classifier_huggingface.pth # Trained model weights
└── processor.pkl                  # Image preprocessing pipeline
```

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip (Python package manager)
```

### Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd Medical-Report-Generation-Based-on-Images
   ```

2. **Create virtual environment**:

   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:

   ```bash
   python app.py
   ```

5. **Access the web interface**:
   ```
   http://localhost:5000
   ```

## 🔧 Configuration

### Model Selection

Edit `classify_function.py` to choose your model backend:

```python
# Option 1: Use custom trained model
model, processor = load_local_model()

# Option 2: Use pre-trained medical model
from classify_function_pretrained import classify_xray_pretrained
```

### API Endpoints

- `GET /` - Main web interface
- `POST /upload` - Image classification endpoint
- `POST /api/analyze` - RESTful API for programmatic access
- `GET /health` - System health check
- `GET /model-info` - Model information and status

## 📊 Model Performance & Validation

### Testing Your Model

Run comprehensive model validation:

```bash
# Test model loading and basic functionality
python classify_function.py

# Test model output variability
python test_model_variability.py

# Debug model architecture and weights
python debug_model.py
```

### Expected Performance Characteristics

**Reliable Model**:

- Different outputs for different inputs
- Balanced predictions (not biased toward one class)
- Reasonable confidence scores (40-95%)
- Consistent preprocessing pipeline

**Problematic Model**:

- Same output for all inputs
- Extreme bias (always predicts one class)
- Overconfident predictions (>98% always)
- Preprocessing issues (zero variance)

## 🩺 Medical Use Cases

### Supported Conditions

- **Normal Chest X-rays**: Clear lung fields, no abnormalities
- **Pneumonia**: Bacterial/viral pneumonia, lung consolidation
- **Future Extensions**: Fractures, tumors, pleural effusion

### Clinical Workflow Integration

```
Patient X-ray → Upload to System → AI Analysis → Radiologist Review → Final Diagnosis
```

### Confidence Score Interpretation

- **90-100%**: High confidence, clear indicators
- **70-89%**: Good confidence, recommend review
- **50-69%**: Moderate confidence, requires expert opinion
- **<50%**: Low confidence, manual analysis required

## ⚠️ Important Limitations

### Technical Limitations

- **Image Quality**: Requires standard chest X-ray format
- **Resolution**: Optimal performance with high-resolution images
- **Positioning**: Standard PA/AP chest positioning preferred
- **File Formats**: Supports JPG, PNG, JPEG

### Medical Limitations

- **Not FDA Approved**: Research and educational use only
- **Requires Expert Review**: AI cannot replace radiologist diagnosis
- **Limited Scope**: Trained for specific conditions only
- **Population Bias**: Model performance may vary across demographics

## 🔍 Troubleshooting

### Common Issues

1. **Model Loading Fails**:

   ```bash
   # Check if model files exist
   ls -la *.pth *.pkl

   # Test model loading
   python debug_model.py
   ```

2. **Same Predictions for All Images**:

   ```bash
   # Test model variability
   python test_model_variability.py
   ```

3. **Memory Issues**:

   ```python
   # Reduce batch size in classify_function.py
   # Use CPU instead of GPU if needed
   device = 'cpu'
   ```

4. **Import Errors**:
   ```bash
   pip install torch torchvision transformers pillow flask
   ```

## 🚀 Deployment Options

### Development Server

```bash
python app.py  # http://localhost:5000
```

### Production Deployment

**Option 1: Gunicorn (Linux/Mac)**

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

**Option 2: Docker**

```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "app.py"]
```

**Option 3: Cloud Platforms**

- Heroku: Direct Git deployment
- Railway: Automatic Flask detection
- Google Cloud Run: Container deployment

## 📈 Future Enhancements

### Planned Features

- **Multi-class Classification**: Fractures, tumors, other conditions
- **Report Generation**: Automated medical report creation
- **DICOM Support**: Medical imaging standard compatibility
- **Batch Processing**: Multiple image analysis
- **Model Ensemble**: Combine multiple AI models for better accuracy

### Research Directions

- **Federated Learning**: Train on distributed medical datasets
- **Explainable AI**: Highlight regions of interest in X-rays
- **Real-time Processing**: WebRTC for live image analysis
- **Mobile Deployment**: Edge computing for mobile devices

## 👥 Contributing

### Development Setup

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and test thoroughly
4. Commit changes: `git commit -m 'Add amazing feature'`
5. Push to branch: `git push origin feature/amazing-feature`
6. Open Pull Request

### Code Standards

- Follow PEP 8 Python style guide
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation for API changes

## 📄 License & Ethics

### License

This project is licensed under the MIT License - see LICENSE file for details.

### Ethical Considerations

- **Patient Privacy**: No patient data stored or transmitted
- **Bias Awareness**: Model performance may vary across populations
- **Professional Responsibility**: Always defer to qualified medical professionals
- **Transparency**: Open source for scrutiny and improvement

## 📞 Support & Contact

### Getting Help

- **Issues**: GitHub Issues for bug reports
- **Discussions**: GitHub Discussions for questions
- **Documentation**: Check README and code comments first

### Medical Collaboration

We welcome collaboration with:

- Medical professionals
- Radiologists and imaging specialists
- Healthcare institutions
- Medical AI researchers

---

**Remember**: This tool is designed to assist, not replace, professional medical judgment. Always consult qualified healthcare providers for medical decisions.
