from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2
import base64

app = Flask(__name__)

# Load the trained model
model_path = 'Final_brain.h5'
try:
    model = load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Function to preprocess the uploaded image
def preprocess_image(image_bytes):
    try:
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        
        # Check if image is grayscale (single-channel)
        if len(img.shape) < 3 or img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        # Resize image to (64, 64) and normalize pixel values
        img = cv2.resize(img, (64, 64))
        img_array = img.astype(np.float32) / 255.0
        
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# Function to predict the class of the uploaded image
def predict_image(image_bytes, model):
    try:
        img_array = preprocess_image(image_bytes)
        if img_array is None:
            return None
        
        # Make prediction using the loaded model
        predictions = model.predict(img_array)
        class_labels = ["glioma", "meningioma", "notumor", "pituitary"]
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_labels[predicted_class_index]
        
        return predicted_class
    except Exception as e:
        print(f"Error predicting image: {e}")
        return None

# Flask routes
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/')
def new():
    return render_template('new.html')

@app.route('/result')
def result():
    # Handle the result logic here
    return render_template('result.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            return "No file uploaded."
        
        # Save the uploaded file to the uploads directory
        filename = 'temp_image.jpg'
        file_path = os.path.join('uploads', filename)
        uploaded_file.save(file_path)
        print(f"Saved uploaded file to: {file_path}")
        
        # Perform prediction
        predicted_class = predict_image(open(file_path, 'rb').read(), model)
        if predicted_class is None:
            return "Prediction failed. Please check the input image."
        
        print(f"Predicted class: {predicted_class}")

        # Read and preprocess the image
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        img_array = preprocess_image(open(file_path, 'rb').read())
        if img_array is None:
            return "Error preprocessing image."

        # Encode the image as a base64 string
        _, buffer = cv2.imencode('.jpg', img)
        img_str = base64.b64encode(buffer).decode()

        # Render result.html with predicted class and base64-encoded image
        return render_template('result.html', img_filename=filename, predicted_class=predicted_class, image=img_str)
    
    except Exception as e:
        return f"Error: {e}"

@app.route('/uploads/<filename>')
def serve_image(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)  # Create the 'uploads' directory if it doesn't exist
    app.run(debug=True)