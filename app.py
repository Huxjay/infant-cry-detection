import os
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
import json
from werkzeug.utils import secure_filename
import tempfile
from pydub import AudioSegment   # ✅ Added

# ================== CONFIG ==================
SAMPLE_RATE = 22050
DURATION = 5
MAX_LENGTH = SAMPLE_RATE * DURATION
N_MFCC = 40
INPUT_SHAPE = (40, 216, 1)

# Initialize Flask app
app = Flask(__name__, 
           template_folder='templates',  # Ensure this folder exists
           static_folder='static')       # Ensure this folder exists

# Load trained model (with error handling)
try:
    MODEL_PATH = "baby_cry_classifier_final.keras"
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Load labels (with error handling)
try:
    with open("label_mapping.json", "r") as f:
        label_mapping = json.load(f)
    id_to_label = {int(v): k for k, v in label_mapping.items()}  # Convert values to int
    print("✅ Label mapping loaded successfully!")
except Exception as e:
    print(f"❌ Error loading label mapping: {e}")
    id_to_label = {}

# ================== AUDIO PREPROCESSING ==================
def preprocess_audio(file_path):
    """Load and preprocess audio file into MFCCs"""
    try:
        # Convert to wav if not already wav
        if not file_path.endswith(".wav"):
            wav_path = file_path + ".wav"
            audio = AudioSegment.from_file(file_path)
            audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)
            audio.export(wav_path, format="wav")
            file_path = wav_path
            print(f"🔄 Converted to WAV: {file_path}")

        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        print(f"✅ Audio loaded: {len(y)} samples, {sr} Hz")

        # Trim/pad to exactly 5 seconds
        target_length = SAMPLE_RATE * DURATION
        if len(y) > target_length:
            y = y[:target_length]
        else:
            y = np.pad(y, (0, target_length - len(y)), 'constant')
        
        print(f"✅ Audio trimmed/padded: {len(y)} samples")

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(
            y=y, 
            sr=sr, 
            n_mfcc=N_MFCC,
            n_fft=2048,
            hop_length=512
        )
        print(f"✅ MFCC shape: {mfccs.shape}")

        # Add dimensions for model input
        mfccs = np.expand_dims(mfccs, axis=-1)  # (40, 216, 1)
        mfccs = np.expand_dims(mfccs, axis=0)   # (1, 40, 216, 1)
        
        return mfccs
        
    except Exception as e:
        print(f"❌ Error in preprocess_audio: {e}")
        return None

# ================== ROUTES ==================
@app.route("/")
def index():
    """Render main page"""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handle audio prediction from microphone"""
    print("📥 Prediction request received")
    
    if "audio" not in request.files:
        print("❌ No audio file in request")
        return jsonify({"error": "No audio file uploaded"}), 400

    file = request.files["audio"]
    if file.filename == '':
        print("❌ Empty filename")
        return jsonify({"error": "No file selected"}), 400

    # Create temporary file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_file:
            file.save(tmp_file.name)
            tmp_path = tmp_file.name
            print(f"✅ Audio saved to temporary file: {tmp_path}")

        # Check if model is loaded
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        # Preprocess audio
        input_data = preprocess_audio(tmp_path)
        if input_data is None:
            return jsonify({"error": "Audio processing failed"}), 500

        print(f"✅ Input data shape: {input_data.shape}")

        # Predict
        preds = model.predict(input_data, verbose=0)
        print(f"✅ Predictions: {preds}")

        class_id = int(np.argmax(preds))
        confidence = float(np.max(preds))
        
        # Get class label
        if class_id in id_to_label:
            class_label = id_to_label[class_id]
        else:
            class_label = f"class_{class_id}"

        print(f"✅ Prediction: {class_label} (confidence: {confidence:.4f})")

        # Clean up temporary file(s)
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        wav_path = tmp_path + ".wav"
        if os.path.exists(wav_path):
            os.unlink(wav_path)

        return jsonify({
            "predicted_class": class_label,
            "confidence": confidence
        })

    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        # Clean up temporary file if it exists
        if 'tmp_path' in locals():
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            wav_path = tmp_path + ".wav"
            if os.path.exists(wav_path):
                os.unlink(wav_path)
        return jsonify({"error": str(e)}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return app.send_static_file(filename)

# Health check endpoint
@app.route('/health')
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

if __name__== "__main__":
    # Create necessary directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("🚀 Starting Flask server...")
    print("📁 Current working directory:", os.getcwd())
    print("📁 Files in directory:", os.listdir('.'))
    
    app.run(debug=True, host='0.0.0.0', port=5000)