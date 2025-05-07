import cv2
import numpy as np
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from tensorflow.keras.models import load_model
import base64
import io
from PIL import Image
import re
import os
import sys
import traceback

# Print debug info
print("Starting emotion detection server...")
print(f"Python version: {sys.version}")
print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")

# Check if model file exists
model_path = './face_model.h5'
if not os.path.exists(model_path):
    print(f"ERROR: Model file not found at {model_path}")
    print(f"Current working directory: {os.getcwd()}")
    print("Please make sure the model file exists.")
    sys.exit(1)

# Load model and labels
try:
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    print("Model loaded successfully!")
    class_names = ['Angry', 'Disgusted', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
except Exception as e:
    print(f"ERROR loading model: {e}")
    traceback.print_exc()
    sys.exit(1)

# Initialize Flask and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'emotion-detection-secret'
socketio = SocketIO(app, cors_allowed_origins='*', logger=True, engineio_logger=True)

# Initialize face cascade
try:
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    print(f"Loading face cascade from {face_cascade_path}")
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        print("ERROR: Face cascade failed to load")
        sys.exit(1)
    print("Face cascade loaded successfully!")
except Exception as e:
    print(f"ERROR loading face cascade: {e}")
    traceback.print_exc()
    sys.exit(1)

# Debug directory setup
# debug_dir = "debug_frames"
# if not os.path.exists(debug_dir):
#     os.makedirs(debug_dir)

# Frame counter for debugging
frame_counter = 0

# Emotion detection function - improved to match original script
def detect_emotion(img_array):
    global frame_counter
    
    try:
        frame_counter += 1
        
        # Save every 50th frame for debugging
        # if frame_counter % 50 == 0:
        #     cv2.imwrite(f"{debug_dir}/frame_{frame_counter}.jpg", cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        
        # Convert to grayscale for face detection (matching original script)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Detect faces using the same parameters as original script
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        
        # If no face detected, return message
        if len(faces) == 0:
            print("No face detected in the frame")
            return "No face detected"
        
        # Process all detected faces, same as original script
        emotions = []
        confidences = []
        
        for (x, y, w, h) in faces:
            # Extract face ROI (Region of Interest)
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize to model input size (48x48) - exactly like original script
            face_roi = cv2.resize(face_roi, (48, 48))
            
            # Save resized face for debugging
            # if frame_counter % 50 == 0:
            #     cv2.imwrite(f"{debug_dir}/face_{frame_counter}_{len(emotions)}.jpg", face_roi)
            
            # Convert to array - match original script processing
            img = np.array(face_roi, dtype=np.float32)
            img = np.expand_dims(img, axis=0)
            
            # Match original script normalization - make sure this matches your model training
            # Note: We try both normalized and unnormalized options
            # if frame_counter % 2 == 0:  # Try without normalization (like original script)
            img_unnormalized = img.copy()
            prediction_unnormalized = model.predict(img_unnormalized)
            predicted_class_unnormalized = np.argmax(prediction_unnormalized)
            emotions.append(class_names[predicted_class_unnormalized])
            confidences.append(prediction_unnormalized[0][predicted_class_unnormalized])
            # else:  # Try with normalization
            #     img_normalized = img / 255.0
            #     prediction_normalized = model.predict(img_normalized)
            #     predicted_class_normalized = np.argmax(prediction_normalized)
            #     emotions.append(class_names[predicted_class_normalized])
            #     confidences.append(prediction_normalized[0][predicted_class_normalized])
            
            # Debug output for every 20th frame
            if frame_counter % 20 == 0:
                print(f"\nPrediction for frame {frame_counter}, face {len(emotions)}:")
                for i, emotion_name in enumerate(class_names):
                #     if frame_counter % 2 == 0:
                        print(f"{emotion_name}: {prediction_unnormalized[0][i]:.4f}")
                    # else:
                        # print(f"{emotion_name}: {prediction_normalized[0][i]:.4f}")
        
        # Get the emotion with highest confidence
        if emotions:
            max_confidence_idx = np.argmax(confidences)
            max_confidence = confidences[max_confidence_idx]
            max_emotion = emotions[max_confidence_idx]
            
            print(f"Best prediction: {max_emotion} with confidence {max_confidence:.4f}")
            
            # Only return if confidence is high enough
            if max_confidence > 0.3:
                return max_emotion
            else:
                return "Low confidence"
        else:
            return "No face"
            
    except Exception as e:
        print(f"Error in detect_emotion: {e}")
        traceback.print_exc()
        return f"Error: {str(e)}"

# WebSocket route for receiving images
@socketio.on('image')
def handle_image(data):
    try:
        # Extract base64 data from the data URL
        base64_data = re.sub('^data:image/.+;base64,', '', data)
        img_data = base64.b64decode(base64_data)
        
        # Convert to PIL Image
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        frame = np.array(img)
        
        # Detect emotion
        emotion = detect_emotion(frame)
        
        # Emit the emotion result back to client
        emit('emotion', {'emotion': emotion})
    except Exception as e:
        print(f"Error processing image: {e}")
        traceback.print_exc()
        emit('error', {'message': str(e)})

@socketio.on('connect')
def handle_connect():
    print('Client connected:', request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected:', request.sid)

@app.route('/')
def index():
    return 'WebSocket server is running. Connect with the Next.js frontend app.'

if __name__ == '__main__':
    print("✅ Server started on http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)