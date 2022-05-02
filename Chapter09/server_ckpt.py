# Important Note:
#
# For your convenience, we have copied the notebook named "Cancer_Detection.ipynb" from Chapter 2 to the
# GitHub folder for this chapter.
#
# Run the "Cancer_Detection.ipynb" notebook on Google colab and download the output epoch=499-step=15499.ckpt
# in the same directory ./lightning_logs/version_0/checkpoints/ before launching this server program.

import torch.nn.functional as functional
import torchvision.transforms as transforms
import torch
import numpy as np

from PIL import Image

from flask import Flask, request, jsonify

from image_classifier import CNNImageClassifier

model = CNNImageClassifier.load_from_checkpoint("./lightning_logs/version_1/checkpoints/epoch=499-step=15499.ckpt")
model.eval()

IMAGE_SIZE = 32
def transform_image(img):
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor()
    ])
    return transform(img).unsqueeze(0)

def get_prediction(img):
    result = model(img)
    return functional.softmax(result, dim=1)[:, 1].tolist()[0]

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    img_file = request.files['image']
    img = Image.open(img_file.stream)
    img = transform_image(img)
    prediction = get_prediction(img)
    if prediction >= 0.5:
        cancer_or_not = "cancer"
    else:
        cancer_or_not = "no_cancer"
    return jsonify({'cancer_or_not': cancer_or_not})

if __name__ == '__main__':
    app.run()
