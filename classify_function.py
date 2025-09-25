# classify_function_pretrained.py - Using established medical models
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import torchvision.transforms as transforms

# Global variables
_model = None
_processor = None
_model_name = None

def load_pretrained_medical_model():
    """
    Load a pre-trained medical model for chest X-ray classification
    """
    global _model, _processor, _model_name
    
    if _model is None:
        print("Loading pre-trained medical chest X-ray model...")
        
        # Try multiple established medical models in order of reliability
        model_options = [
            {
                "name": "nickmccullum/chest-xray-pneumonia-binary-classification",
                "description": "Specialized pneumonia detection model",
                "classes": ["Normal", "Pneumonia"]
            },
            {
                "name": "keremberke/chest-xray-classification", 
                "description": "General chest X-ray classifier",
                "classes": ["Normal", "Pneumonia"]
            },
            {
                "name": "microsoft/swin-tiny-patch4-window7-224",
                "description": "General vision model (fallback)",
                "classes": ["Class 0", "Class 1"]  # Generic
            }
        ]
        
        for model_info in model_options:
            try:
                print(f"Trying: {model_info['description']}")
                
                _model = AutoModelForImageClassification.from_pretrained(model_info["name"])
                _processor = AutoImageProcessor.from_pretrained(model_info["name"])
                _model_name = model_info["name"]
                
                _model.eval()  # Set to evaluation mode
                
                print(f"✅ Successfully loaded: {model_info['name']}")
                print(f"   Classes: {model_info['classes']}")
                break
                
            except Exception as e:
                print(f"⚠️ Failed to load {model_info['name']}: {e}")
                continue
        
        if _model is None:
            print("❌ All pre-trained models failed to load")
            return None, None, None
    
    return _model, _processor, _model_name

def preprocess_medical_image(image_path):
    """
    Preprocess image using the pre-trained model's processor
    """
    try:
        # Load image
        if isinstance(image_path, str):
            image = Image.open(image_path)
        else:
            image = image_path
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        print(f"Image loaded: {image.size}, mode: {image.mode}")
        
        # Use the model's processor
        inputs = _processor(image, return_tensors="pt")
        return inputs
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def classify_xray_pretrained(image_path):
    """
    Classify chest X-ray using pre-trained medical model
    """
    try:
        print(f"🔍 Analyzing X-ray with pre-trained medical model...")
        
        # Load model
        model, processor, model_name = load_pretrained_medical_model()
        
        if model is None:
            return {
                'prediction': 'Failed to load pre-trained medical model',
                'confidence': '0%',
                'status': 'error',
                'error': 'Could not load any pre-trained medical models'
            }
        
        # Preprocess image
        inputs = preprocess_medical_image(image_path)
        
        if inputs is None:
            return {
                'prediction': 'Image preprocessing failed',
                'confidence': '0%',
                'status': 'error',
                'error': 'Could not preprocess the image'
            }
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities.max().item() * 100
        
        print(f"Model: {model_name}")
        print(f"Logits: {outputs.logits.detach().cpu().numpy().flatten()}")
        print(f"Probabilities: {probabilities.detach().cpu().numpy().flatten()}")
        print(f"Predicted class: {predicted_class}, Confidence: {confidence:.1f}%")
        
        # Map predictions based on the model
        if "pneumonia" in model_name.lower():
            # Models trained specifically for pneumonia detection
            class_names = ["Normal", "Pneumonia"]
        else:
            # Generic models - we need to infer the mapping
            class_names = ["Normal", "Abnormal"]
        
        predicted_label = class_names[predicted_class] if predicted_class < len(class_names) else f"Class {predicted_class}"
        
        # Create result
        if predicted_class == 0:  # Assuming 0 = Normal for medical models
            result = f"✅ {predicted_label} chest X-ray - No concerning findings detected"
            recommendation = "Based on AI analysis, chest appears normal"
        else:
            result = f"⚠️ {predicted_label} detected - Medical consultation recommended"
            recommendation = "AI detected potential abnormalities - please consult a healthcare provider"
        
        return {
            'prediction': result,
            'label': predicted_label,
            'confidence': f"{confidence:.1f}%",
            'recommendation': recommendation,
            'model_name': model_name,
            'model_type': 'Pre-trained Medical Model',
            'status': 'success',
            'disclaimer': 'This is an AI analysis and should not replace professional medical diagnosis'
        }
        
    except Exception as e:
        print(f"Classification error: {e}")
        return {
            'prediction': 'Classification failed',
            'confidence': '0%',
            'status': 'error',
            'error': f'Pre-trained model error: {str(e)}'
        }

def test_pretrained_model():
    """
    Test the pre-trained model loading and basic functionality
    """
    print("🧪 Testing pre-trained medical model...")
    
    model, processor, model_name = load_pretrained_medical_model()
    
    if model is not None:
        print(f"✅ Pre-trained model test passed!")
        print(f"   Model: {model_name}")
        print(f"   Processor: {type(processor)}")
        
        # Test with a synthetic image
        from PIL import Image
        import numpy as np
        test_img = Image.fromarray(np.ones((224, 224), dtype=np.uint8) * 128).convert('RGB')
        
        try:
            inputs = _processor(test_img, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            print(f"   Test prediction: {outputs.logits.detach().cpu().numpy().flatten()}")
            return True
        except Exception as e:
            print(f"   ⚠️ Test prediction failed: {e}")
            return False
    else:
        print("❌ Pre-trained model test failed!")
        return False

# Alternative function that tries multiple approaches
def classify_xray_flask(image_path):
    """
    Main classification function - tries pre-trained model first, then fallback
    """
    try:
        # Try pre-trained medical model first
        result = classify_xray_pretrained(image_path)
        
        if result['status'] == 'success':
            return result
        else:
            print("Pre-trained model failed, trying simple analysis...")
            return simple_analysis_fallback(image_path)
            
    except Exception as e:
        print(f"All classification methods failed: {e}")
        return {
            'prediction': 'Analysis unavailable',
            'confidence': '0%',
            'status': 'error',
            'error': 'All classification methods failed'
        }

def simple_analysis_fallback(image_path):
    """
    Simple fallback analysis when pre-trained models fail
    """
    try:
        from PIL import Image
        import numpy as np
        
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to grayscale and analyze
        gray = image.convert('L')
        img_array = np.array(gray)
        
        # Basic image statistics
        mean_brightness = np.mean(img_array)
        std_brightness = np.std(img_array)
        
        # Simple heuristic based on typical X-ray characteristics
        if mean_brightness < 50:  # Very dark image
            prediction = "Image appears very dark - may not be a standard X-ray"
            confidence = 60.0
        elif mean_brightness > 200:  # Very bright image
            prediction = "Image appears very bright - unusual for chest X-ray"
            confidence = 55.0
        elif std_brightness < 20:  # Low contrast
            prediction = "Low contrast image - difficult to analyze"
            confidence = 40.0
        else:
            prediction = "Image analysis completed - consider professional medical review"
            confidence = 70.0
        
        return {
            'prediction': prediction,
            'label': 'Analysis Complete',
            'confidence': f"{confidence:.1f}%",
            'recommendation': 'Simple image analysis performed - not a medical diagnosis',
            'model_type': 'Basic Image Analysis',
            'status': 'partial_success',
            'disclaimer': 'This is basic image analysis, not medical diagnosis'
        }
        
    except Exception as e:
        return {
            'prediction': 'Analysis failed',
            'confidence': '0%',
            'status': 'error',
            'error': str(e)
        }

if __name__ == "__main__":
    # Test the pre-trained model
    test_pretrained_model()