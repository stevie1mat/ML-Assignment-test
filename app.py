from flask import Flask, request, jsonify, render_template_string
from PIL import Image, ImageOps
import requests
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Function to recognize handwriting using the model
def recognize_handwriting(image, model):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    recognized_text = np.argmax(prediction, axis=1)[0]
    return recognized_text

# HTML template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Handwriting Recognition</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            background: linear-gradient(to right, #ffecd2, #fcb69f);
            text-align: center;
        }
        .container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        h1 {
            margin-top: 0;
        }
        input[type="url"] {
            padding: 10px;
            width: 80%;
            margin-bottom: 10px;
            border-radius: 5px;
            border: 1px solid #ccc;
        }
        button {
            padding: 10px 20px;
            background-color: #fcb69f;
            border: none;
            border-radius: 5px;
            color: white;
            cursor: pointer;
        }
        button:hover {
            background-color: #ff7e5f;
        }
        .result {
            margin-top: 20px;
        }
        img {
            max-width: 100%;
            max-height: 300px;
            border-radius: 10px;
            margin-top: 10px;
        }
        .author {
            margin-top: 20px;
            font-style: italic;
            color: #888;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Handwriting Recognition</h1>
        <form method="post">
            <input type="url" name="url" placeholder="Enter image URL" required>
            <button type="submit">Submit</button>
        </form>
        {% if recognized_text is not none %}
            <div class="result">
                <h2>Recognized Text:</h2>
                <p>{{ recognized_text }}</p>
                <h2>Image:</h2>
                <img src="{{ image_url }}" alt="Handwritten Image">
            </div>
        {% endif %}
        {% if error %}
            <div class="result">
                <h2>Error:</h2>
                <p>{{ error }}</p>
            </div>
        {% endif %}
        <div class="author">
            <p>Author: Steven Mathew</p>
        </div>
    </div>
</body>
</html>
'''

# Preprocess the image for the model
def preprocess_image(image):
    # Invert image colors (MNIST dataset uses white digits on black background)
    image = ImageOps.invert(image)
    # Resize the image to 28x28 pixels
    image = image.resize((28, 28))
    # Normalize the image
    image_array = np.array(image)
    image_array = image_array / 255.0
    # Add batch dimension and channel dimension for grayscale
    image_array = np.expand_dims(image_array, axis=0)
    image_array = np.expand_dims(image_array, axis=-1)
    return image_array

# Load a pre-trained model (for this example, we use a simple CNN model trained on MNIST)
def load_handwriting_model():
    model_url = "https://stevenmathew.dev/demo/mnist_model.h5"
    model_path = tf.keras.utils.get_file("mnist_model.h5", model_url)
    model = load_model(model_path)
    return model

# Load the model once at startup
model = load_handwriting_model()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    recognized_text = None
    error = None
    image_url = None
    if request.method == 'POST':
        url = request.form['url']
        image_url = url  # Save the image URL to pass to the template
        if url:
            try:
                response = requests.get(url)
                response.raise_for_status()  # Check if the request was successful
                image = Image.open(io.BytesIO(response.content)).convert("L")  # Convert image to grayscale
                recognized_text = recognize_handwriting(image, model)
            except requests.exceptions.RequestException as e:
                error = f"Failed to fetch image: {e}"
            except IOError as e:
                error = f"Invalid image format: {e}"
    return render_template_string(HTML_TEMPLATE, recognized_text=recognized_text, image_url=image_url, error=error)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'url' in request.json:
        url = request.json['url']
        try:
            response = requests.get(url)
            response.raise_for_status()  # Check if the request was successful
            image = Image.open(io.BytesIO(response.content)).convert("L")  # Convert image to grayscale
            recognized_text = recognize_handwriting(image, model)
            return jsonify({'recognized_text': recognized_text})
        except requests.exceptions.RequestException as e:
            return jsonify({'error': f"Failed to fetch image: {e}"}), 400
        except IOError as e:
            return jsonify({'error': f"Invalid image format: {e}"}), 400
    else:
        return jsonify({'error': 'URL is required'}), 400
