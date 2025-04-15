import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
model = YOLO('app/my_model.pt')
model.eval()

def get_prediction(img_array):
    results = model(img_array)  # includes NMS
    for r in results:

        annotator = Annotator(img_array)

        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            c = box.cls
            annotator.box_label(b, model.names[int(c)])

    image = annotator.result()
    return image


class PredictCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()

        result = get_prediction(frame)
        ret, jpeg = cv2.imencode('.jpg', result)

        return jpeg.tobytes()

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()

        # DO WHAT YOU WANT WITH TENSORFLOW / KERAS AND OPENCV

        ret, jpeg = cv2.imencode('.jpg', frame)

        return jpeg.tobytes()