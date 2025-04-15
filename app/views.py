import io
from flask import Blueprint, render_template, request, redirect, Response, jsonify
from PIL import Image
from .camera import VideoCamera, PredictCamera
import datetime

import torch
from ultralytics import YOLO

DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"

views = Blueprint('views', __name__)

model = YOLO('app/my_model.pt')
model.eval()
video_stream = VideoCamera()
prediction_stream = PredictCamera()

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

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@views.route('/video_feed')
def video_feed():
     return Response(gen(video_stream),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@views.route("/prediction_feed")
def predict_feed():
    return Response(gen(prediction_stream),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def get_prediction(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    imgs = [img]  # batched list of images

# Inference
    results = model(imgs)  # includes NMS
    return results