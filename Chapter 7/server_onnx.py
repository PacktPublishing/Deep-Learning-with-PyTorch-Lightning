import onnxruntime
import numpy as np

import torchvision.transforms as transforms
from PIL import Image

from flask import Flask, request, jsonify

session = onnxruntime.InferenceSession("model.onnx", None)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

IMAGE_SIZE = 64
def transform_image(img):
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor()
    ])
    return transform(img).unsqueeze(0)

def get_prediction(img):
    result = session.run([output_name], {input_name: img})
    return int(np.argmax(np.array(result).squeeze(), axis=0))

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    img_file = request.files['image']
    img = Image.open(img_file.stream)
    img = transform_image(img)
    prediction = get_prediction(img.numpy())
    if prediction >= 0.5:
        cat_or_dog = "dog"
    else:
        cat_or_dog = "cat"
    return jsonify({'cat_or_dog': cat_or_dog})
 
if __name__ == '__main__':
    app.run()
