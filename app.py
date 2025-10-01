# Flask Chest X-Ray Classifier - Updated for ViT Model
# app.py

from flask import Flask, request, render_template, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import os
import time
from PIL import Image
import io
import base64

# Import your classification function
from classify_function import classify_xray_flask

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page with upload form"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and classification"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Save uploaded file temporarily
            filename = secure_filename(file.filename)
            timestamp = str(int(time.time()))
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            print(f"🔬 Analyzing X-ray with ViT model: {filename}")
            
            # Classify the image
            start_time = time.time()
            result = classify_xray_flask(filepath)
            processing_time = time.time() - start_time
            
            # Add processing time to result
            result['processing_time'] = round(processing_time, 2)
            result['filename'] = filename
            
            print(f"✅ Analysis complete: {result['prediction']} ({result['confidence']})")
            
            # Clean up - remove uploaded file
            try:
                os.remove(filepath)
            except:
                pass
            
            return jsonify(result)
        
        else:
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG files.'}), 400
    
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for programmatic access"""
    try:
        # Handle JSON data with base64 encoded image
        if request.is_json:
            data = request.get_json()
            if 'image' not in data:
                return jsonify({'error': 'No image data provided'}), 400
            
            # Decode base64 image
            image_data = data['image']
            if image_data.startswith('data:'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            print("🔬 API: Analyzing X-ray with ViT model")
            
            # Classify
            result = classify_xray_flask(image)
            return jsonify(result)
        
        # Handle form data upload
        else:
            return upload_file()
    
    except Exception as e:
        print(f"❌ API error: {e}")
        return jsonify({'error': f'API error: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'chest-xray-classifier',
        'model_type': 'Vision Transformer (ViT)',
        'version': '2.0.0'
    })

@app.route('/model-info')
def model_info():
    """Get information about the loaded model"""
    try:
        from classify_function import _model, _processor
        
        model_loaded = _model is not None
        processor_loaded = _processor is not None
        
        return jsonify({
            'model_loaded': model_loaded,
            'processor_loaded': processor_loaded,
            'model_type': 'Vision Transformer (ViT)' if model_loaded else 'Not loaded',
            'architecture': 'google/vit-base-patch16-224-in21k',
            'classes': ['Normal', 'Pneumonia']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Get environment from ENV variable (defaults to 'dev')
    env = os.environ.get('ENV', 'dev').lower()
    
    if env == 'dev':
        # Development mode - local
        print("🚀 Starting Chest X-Ray Classifier Flask App (DEV MODE)")
        print("🤖 Model: Vision Transformer (ViT)")
        print("📁 Upload folder:", app.config['UPLOAD_FOLDER'])
        print("🔗 Access at: http://localhost:5000")
        print("📊 Health check: http://localhost:5000/health")
        print("🧠 Model info: http://localhost:5000/model-info")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        # Production mode - Railway/cloud
        port = int(os.environ.get('PORT', 5000))
        print("🚀 Starting Chest X-Ray Classifier Flask App (PRODUCTION MODE)")
        print(f"🌐 Running on port: {port}")
        app.run(debug=False, host='0.0.0.0', port=port)