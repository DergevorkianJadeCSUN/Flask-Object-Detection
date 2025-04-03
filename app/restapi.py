import argparse
import io
from PIL import Image

import torch
from flask import Flask, request

restapi = Flask(__name__)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

URL = "/v1/object-detection/yolov5"

@restapi.route(URL, methods=["POST"])
def predict():
    if not request.method == "POST":
        return

    if request.files.get("image"):
        image_file = request.files["image"]
        image_bytes = image_file.read()
        results = get_prediction(image_bytes)
        return results


def get_prediction(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    imgs = [img]  # batched list of images

# Inference
    results = model(imgs, size=640)  # includes NMS
    return results