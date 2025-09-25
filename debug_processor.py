import pickle
import torch

def debug_processor():
    """
    Debug what's inside your processor.pkl
    """
    print("🔍 Debugging processor.pkl...")
    
    try:
        with open('processor.pkl', 'rb') as f:
            processor = pickle.load(f)
        
        print(f"✅ Processor loaded successfully")
        print(f"📋 Type: {type(processor)}")
        
        # Check attributes
        if hasattr(processor, '__dict__'):
            print(f"📝 Attributes: {list(processor.__dict__.keys())}")
            
            for key, value in processor.__dict__.items():
                print(f"   {key}: {value}")
        
        # Check if it has normalization values
        if hasattr(processor, 'image_mean'):
            print(f"🎯 Image mean: {processor.image_mean}")
        if hasattr(processor, 'image_std'):
            print(f"🎯 Image std: {processor.image_std}")
        if hasattr(processor, 'size'):
            print(f"🎯 Size: {processor.size}")
        if hasattr(processor, 'do_normalize'):
            print(f"🎯 Do normalize: {processor.do_normalize}")
        if hasattr(processor, 'do_resize'):
            print(f"🎯 Do resize: {processor.do_resize}")
            
        # Test the processor
        print("\n🧪 Testing processor with sample image...")
        from PIL import Image
        import numpy as np
        
        # Create a test image (simulating a normal X-ray - darker overall)
        test_img = Image.fromarray(np.ones((224, 224), dtype=np.uint8) * 50).convert('RGB')
        
        try:
            result = processor(test_img, return_tensors="pt")
            print(f"✅ Processor test successful!")
            print(f"   Output type: {type(result)}")
            print(f"   Keys: {list(result.keys()) if hasattr(result, 'keys') else 'Not a dict'}")
            
            if 'pixel_values' in result:
                pixel_values = result['pixel_values']
                print(f"   Shape: {pixel_values.shape}")
                print(f"   Mean: {pixel_values.mean():.4f}")
                print(f"   Std: {pixel_values.std():.4f}")
                print(f"   Min: {pixel_values.min():.4f}")
                print(f"   Max: {pixel_values.max():.4f}")
                
                # Check if this looks like proper normalization
                if abs(pixel_values.mean()) < 0.1 and 0.8 < pixel_values.std() < 1.2:
                    print("   ✅ Normalization looks correct (mean~0, std~1)")
                else:
                    print("   ⚠️ Normalization might be wrong!")
                    
        except Exception as e:
            print(f"   ❌ Processor test failed: {e}")
    
    except Exception as e:
        print(f"❌ Failed to load processor: {e}")

def debug_model_checkpoint():
    """
    Check what training settings were saved with your model
    """
    print("\n🔍 Debugging model checkpoint...")
    
    try:
        checkpoint = torch.load('xray_classifier_huggingface.pth', map_location='cpu')
        
        if isinstance(checkpoint, dict):
            print(f"📋 Checkpoint keys: {list(checkpoint.keys())}")
            
            # Look for training configuration
            if 'model_config' in checkpoint:
                config = checkpoint['model_config']
                print(f"⚙️ Model config: {config}")
                
            if 'class_names' in checkpoint:
                classes = checkpoint['class_names']
                print(f"🏷️ Classes: {classes}")
                
                # CHECK THIS: Are the class labels in the right order?
                if classes == ['Normal', 'Pneumonia']:
                    print("   ✅ Class order: 0=Normal, 1=Pneumonia")
                elif classes == ['Pneumonia', 'Normal']:
                    print("   ⚠️ Class order: 0=Pneumonia, 1=Normal (REVERSED!)")
                else:
                    print(f"   ❓ Unusual class order: {classes}")
            
            if 'architecture' in checkpoint:
                arch = checkpoint['architecture']
                print(f"🏗️ Architecture: {arch}")
                
            # Look for training stats that might indicate issues
            if 'train_accuracy' in checkpoint:
                print(f"📊 Training accuracy: {checkpoint['train_accuracy']}")
            if 'val_accuracy' in checkpoint:
                print(f"📊 Validation accuracy: {checkpoint['val_accuracy']}")
                
    except Exception as e:
        print(f"❌ Failed to debug checkpoint: {e}")

if __name__ == "__main__":
    debug_processor()
    debug_model_checkpoint()