from flask import Flask, render_template, request, send_from_directory, redirect, url_for, session
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)
app.secret_key = 'b2a4cd9d57a5a5ae1478f17e24802b1297c3b497a1820aa9dc8841a7909b2'  # Required for session

# Load the trained model
model = load_model('model.h5')

# Class labels
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# Upload folder config
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Prediction logic
def predict_tumor(image_path):
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    label = class_labels[predicted_index]
    if label == 'notumor':
        return "No Tumor", confidence
    return f"Tumor: {label}", confidence

# Home page (GET only)
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', result=None)

# Handle image upload and prediction
@app.route('/', methods=['POST'])
def upload():
    file = request.files['file']
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        result, confidence = predict_tumor(file_path)

        # Store in session temporarily
        session['result'] = result
        session['confidence'] = f"{confidence*100:.2f}"
        session['filename'] = file.filename

        return redirect(url_for('result_page'))

    return redirect(url_for('index'))

# Result page (GET only, clears session after showing)
@app.route('/result', methods=['GET'])
def result_page():
    result = session.pop('result', None)
    confidence = session.pop('confidence', None)
    filename = session.pop('filename', None)

    file_path = f"/uploads/{filename}" if filename else None

    return render_template('index.html', result=result, confidence=confidence, file_path=file_path)

# Serve uploaded image files
@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
