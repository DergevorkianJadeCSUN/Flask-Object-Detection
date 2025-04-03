import io
from flask import Blueprint, render_template, request, redirect
from PIL import Image
import datetime

import torch

DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"

views = Blueprint('views', __name__)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
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

        results.render()
        now = datetime.datetime.now().strftime(DATETIME_FORMAT)
        img_name = f"static/images/{now}.png"
        #results.ims[0].save(img_name, "app/static/images")
        Image.fromarray(results.ims[0], 'RGB').save("app/" + img_name)
        return redirect(img_name)
    return render_template('index.html')


def get_prediction(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    imgs = [img]  # batched list of images

# Inference
    results = model(imgs, size=640)  # includes NMS
    return results