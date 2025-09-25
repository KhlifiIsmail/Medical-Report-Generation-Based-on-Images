from classify_function import classify_xray_flask, load_local_model, preprocess_image
from PIL import Image
import numpy as np
import torch
import tempfile
import os

def create_test_images():
    """
    Create drastically different test images
    """
    print("🎨 Creating test images...")
    
    # Image 1: All black (should be very different from white)
    black_img = Image.fromarray(np.zeros((224, 224), dtype=np.uint8)).convert('RGB')
    
    # Image 2: All white
    white_img = Image.fromarray(np.ones((224, 224), dtype=np.uint8) * 255).convert('RGB')
    
    # Image 3: Checkerboard pattern
    checker = np.zeros((224, 224), dtype=np.uint8)
    checker[::2, ::2] = 255
    checker[1::2, 1::2] = 255
    checker_img = Image.fromarray(checker).convert('RGB')
    
    # Image 4: Gradient
    gradient = np.linspace(0, 255, 224*224).reshape(224, 224).astype(np.uint8)
    gradient_img = Image.fromarray(gradient).convert('RGB')
    
    return [
        ("Black", black_img),
        ("White", white_img), 
        ("Checkerboard", checker_img),
        ("Gradient", gradient_img)
    ]

def test_model_outputs():
    """
    Test if the model gives different outputs for drastically different inputs
    """
    print("🧪 Testing model output variability...")
    print("="*60)
    
    # Load model once
    model, processor = load_local_model()
    if model is None:
        print("❌ Failed to load model!")
        return
    
    test_images = create_test_images()
    results = []
    
    for name, img in test_images:
        print(f"\n🖼️ Testing {name} image...")
        
        # Save temporarily and test
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
            
        # Save image (file is now closed, so Windows can access it)
        img.save(tmp_path)
        
        try:
            # Test preprocessing
            pixel_values = preprocess_image(tmp_path)
            if pixel_values is None:
                print(f"   ❌ Preprocessing failed")
                continue
                
            print(f"   📊 Preprocessed - Mean: {pixel_values.mean():.4f}, Std: {pixel_values.std():.4f}")
            
            # Test model directly
            with torch.no_grad():
                logits = model(pixel_values)
                probabilities = torch.softmax(logits, dim=1)
                
            print(f"   🧠 Logits: {logits.detach().cpu().numpy().flatten()}")
            print(f"   📈 Probabilities: {probabilities.detach().cpu().numpy().flatten()}")
            
            results.append({
                'name': name,
                'logits': logits.detach().cpu().numpy().flatten(),
                'probs': probabilities.detach().cpu().numpy().flatten()
            })
            
        finally:
            # Clean up - try multiple times if needed
            try:
                os.unlink(tmp_path)
            except PermissionError:
                import time
                time.sleep(0.1)
                try:
                    os.unlink(tmp_path)
                except:
                    print(f"   ⚠️ Could not delete temp file: {tmp_path}")
                    pass
    
    # Analyze results
    print("\n" + "="*60)
    print("📊 ANALYSIS:")
    
    if len(results) < 2:
        print("❌ Not enough results to analyze")
        return
        
    # Check if all logits are identical or very similar
    first_logits = results[0]['logits']
    all_identical = True
    
    for result in results[1:]:
        diff = np.abs(result['logits'] - first_logits).max()
        print(f"   {result['name']} vs {results[0]['name']}: Max difference = {diff:.6f}")
        if diff > 0.001:  # Allow for tiny floating point differences
            all_identical = False
    
    if all_identical:
        print("\n🚨 CRITICAL ISSUE: All outputs are identical!")
        print("   This means:")
        print("   1. ❌ Model weights are not loaded correctly, OR")
        print("   2. ❌ Model is broken/frozen, OR") 
        print("   3. ❌ There's a bug in the forward pass")
        
        # Check if the model has learnable parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n🔍 Model info:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        # Check if parameters look random or trained
        first_layer = None
        for name, param in model.named_parameters():
            if 'weight' in name:
                first_layer = param
                break
                
        if first_layer is not None:
            param_std = first_layer.std().item()
            param_mean = first_layer.mean().item()
            print(f"   First layer stats - Mean: {param_mean:.6f}, Std: {param_std:.6f}")
            
            if param_std < 0.001:
                print("   ⚠️ Parameters have very low variance - might be zeros or not loaded!")
            elif param_std > 2.0:
                print("   ⚠️ Parameters have very high variance - might be random!")
            else:
                print("   ✅ Parameters look reasonable")
        
    else:
        print("\n✅ Model outputs vary correctly with different inputs")
        print("   The issue might be elsewhere (class labels, training data, etc.)")

def debug_weight_loading():
    """
    Debug if the model weights are actually loaded
    """
    print("\n🔍 Debugging weight loading...")
    
    # Load checkpoint directly
    checkpoint = torch.load('xray_classifier_huggingface.pth', map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        
        print(f"📋 Checkpoint has {len(state_dict)} parameters")
        
        # Look at a few parameters
        param_names = list(state_dict.keys())[:5]
        print("🔍 Sample parameters from checkpoint:")
        
        for name in param_names:
            param = state_dict[name]
            print(f"   {name}: shape={param.shape}, mean={param.mean():.6f}, std={param.std():.6f}")
            
        # Load model and compare
        model, _ = load_local_model()
        
        print("\n🔍 Same parameters in loaded model:")
        for name in param_names:
            if hasattr(model, name.split('.')[0]):
                # Try to get the parameter from loaded model
                try:
                    model_param = dict(model.named_parameters())[name]
                    print(f"   {name}: mean={model_param.mean():.6f}, std={model_param.std():.6f}")
                    
                    # Check if they match
                    if name in state_dict:
                        diff = (model_param - state_dict[name]).abs().max().item()
                        if diff < 1e-6:
                            print(f"      ✅ Matches checkpoint")
                        else:
                            print(f"      ❌ Different from checkpoint (diff: {diff:.6f})")
                            
                except:
                    print(f"   {name}: Not found in loaded model")

if __name__ == "__main__":
    print("🩺 MEDICAL AI MODEL DEBUG")
    print("Testing if model gives consistent outputs for different inputs")
    print("(This should NOT happen - model should vary with input)")
    print("="*60)
    
    test_model_outputs()
    debug_weight_loading()