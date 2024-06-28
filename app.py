from flask import Flask, request, render_template_string, redirect, url_for
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
from io import BytesIO

app = Flask(__name__)

# Load the trained MNIST model
model = tf.keras.models.load_model('mnist_model.h5')

def preprocess_image(image):
    # Convert the image to grayscale and resize it to 28x28
    image = image.convert('L')
    image = image.resize((28, 28))
    # Convert the image to a numpy array and normalize
    image = np.array(image) / 255.0
    # Expand dimensions to match the input shape for the model
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form['url']
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            digit = np.argmax(prediction)
            return render_template_string(html, url=url, digit=digit)
        except Exception as e:
            return str(e)
    return render_template_string(html)

html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MNIST Digit Predictor</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            padding: 50px;
        }
        input[type=text] {
            width: 80%;
            padding: 10px;
            margin: 10px 0;
            box-sizing: border-box;
        }
        input[type=submit] {
            padding: 10px 20px;
            margin: 10px 0;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
        }
        input[type=submit]:hover {
            background-color: #45a049;
        }
        .result {
            margin-top: 20px;
            font-size: 24px;
        }
    </style>
</head>
<body>
    <h1>MNIST Digit Predictor</h1>
    <form method="post">
        <input type="text" name="url" placeholder="Enter the URL of a handwritten digit image" required>
        <input type="submit" value="Predict">
    </form>
    {% if url %}
        <div class="result">
            <p><strong>Image URL:</strong> {{ url }}</p>
            <p><strong>Predicted Digit:</strong> {{ digit }}</p>
            <img src="{{ url }}" alt="Handwritten Digit Image" style="width: 200px; height: 200px;">
        </div>
    {% endif %}
</body>
</html>
'''
if __name__ == '__main__':
    app.run()
