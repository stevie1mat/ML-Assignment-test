from flask import Flask, request, render_template_string, redirect, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import requests
from io import BytesIO

app = Flask(__name__)

# Load the trained model
model = load_model('fashion_mnist_model.h5')

@app.route('/')
def index():
    with open('index.html', 'r') as file:
        index_html = file.read()
    return render_template_string(index_html)

@app.route('/predict', methods=['POST'])
def predict():
    image_url = request.form.get('image_url')

    if not image_url:
        return redirect(url_for('index'))

    try:
        # Fetch the image from the URL
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))

        # Convert the image to grayscale
        image = image.convert('L')

        # Resize the image to 28x28 pixels
        image = image.resize((28, 28))

        # Convert the image to a numpy array
        image_array = np.array(image)

        # Normalize the image
        image_array = image_array / 255.0

        # Reshape the image for the model
        image_array = image_array.reshape(1, 28, 28)

        # Make a prediction
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Render the result in the template
        with open('index.html', 'r') as file:
            index_html = file.read()
        return render_template_string(index_html, prediction=predicted_class)

    except Exception as e:
        return str(e)
if __name__ == '__main__':
    app.run()
