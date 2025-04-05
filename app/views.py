import io
from flask import Blueprint, render_template, request, redirect
from PIL import Image
import datetime

import torch
from ultralytics import YOLO

DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"

views = Blueprint('views', __name__)

model = YOLO('app/my_model.pt')
model.eval()

@views.route('/', methods=['GET','POST'])
def predict():
    if request.method =="POST":
        if "file" not in request.files:
            redirect(request.url)
        file = request.files["file"]
        if not file:
            return

        img_bytes = file.read()
        results = get_prediction(img_bytes)
        now = datetime.datetime.now().strftime(DATETIME_FORMAT)
        img_name = f"static/images/{now}.png"
        #results.ims[0].save(img_name, "app/static/images")
        #Image.fromarray(results[0].numpy(), 'RGB').save("app/" + img_name)
        results[0].save("app/" + img_name)
        return redirect(img_name)
    return render_template('index.html')


def get_prediction(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    imgs = [img]  # batched list of images

# Inference
    results = model(imgs)  # includes NMS
    return results