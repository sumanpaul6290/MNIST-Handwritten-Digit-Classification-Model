from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your trained model
model = load_model('bestmodel.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    image = np.array(data['image'])  # Expecting a 28x28 image array
    image = image.reshape(1, 28, 28, 1)  # Reshape for the model
    prediction = model.predict(image)
    digit = np.argmax(prediction)
    return jsonify({'digit': int(digit)})
