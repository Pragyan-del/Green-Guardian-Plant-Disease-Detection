from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load model (placeholder)
model = load_model('model.h5')

# Dummy classes
classes = ['Healthy', 'Powdery Mildew', 'Leaf Spot', 'Rust']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded!"
    file = request.files['file']
    if file.filename == '':
        return "No file selected!"
    file_path = os.path.join('static', file.filename)
    file.save(file_path)

    img = image.load_img(file_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    result = model.predict(img_array)
    predicted_class = classes[np.argmax(result)]
    return f"Prediction: {predicted_class}"

if __name__ == '__main__':
    app.run(debug=True)
