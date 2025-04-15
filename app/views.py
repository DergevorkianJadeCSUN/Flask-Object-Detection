from flask import Blueprint, render_template, Response
from .camera import VideoCamera, PredictCamera

views = Blueprint('views', __name__)

video_stream = VideoCamera()
prediction_stream = PredictCamera()

@views.route('/', methods=['GET','POST'])
def home():
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
