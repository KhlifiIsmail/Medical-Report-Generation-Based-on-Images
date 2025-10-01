# classify_function.py
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image

_model = None
_processor = None
_model_name = None

def load_pretrained_medical_model():
    global _model, _processor, _model_name
    
    if _model is None:
        print("Loading pre-trained medical chest X-ray model...")
        
        try:
            _model_name = "lxyuan/vit-xray-pneumonia-classification"
            print(f"Loading: {_model_name}")
            
            _model = AutoModelForImageClassification.from_pretrained(_model_name)
            _processor = AutoImageProcessor.from_pretrained(_model_name)
            _model.eval()
            
            print(f"Successfully loaded model!")
            return _model, _processor, _model_name
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            return None, None, None
    
    return _model, _processor, _model_name

def preprocess_medical_image(image_path):
    try:
        if isinstance(image_path, str):
            image = Image.open(image_path)
        else:
            image = image_path
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        inputs = _processor(image, return_tensors="pt")
        return inputs
        
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None

def classify_xray_flask(image_path):
    try:
        model, processor, model_name = load_pretrained_medical_model()
        
        if model is None:
            return {
                'prediction': 'Model loading failed',
                'confidence': '0%',
                'status': 'error'
            }
        
        inputs = preprocess_medical_image(image_path)
        if inputs is None:
            return {
                'prediction': 'Image processing failed',
                'confidence': '0%',
                'status': 'error'
            }
        
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities.max().item() * 100
        
        id2label = model.config.id2label
        predicted_label = id2label.get(predicted_class, f"Class {predicted_class}")
        
        if "NORMAL" in predicted_label.upper():
            result = f"Normal chest X-ray - No concerning findings detected"
        else:
            result = f"Pneumonia detected - Medical consultation recommended"
        
        return {
            'prediction': result,
            'label': predicted_label,
            'confidence': f"{confidence:.1f}%",
            'status': 'success'
        }
        
    except Exception as e:
        print(f"Classification error: {e}")
        return {
            'prediction': f'Error: {str(e)}',
            'confidence': '0%',
            'status': 'error'
        }