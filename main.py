from flask import Flask, request, jsonify
import numpy
from utils import transform_image, get_prediction
import cv2
from flask_cors import CORS, cross_origin


app = Flask(__name__)


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff'}


def allowed_file(filename):
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        filestr = request.files['file'].read()
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})

        try:
            npimg = numpy.fromstring(filestr, numpy.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            transformed = transform_image(img)

            prediction = get_prediction(transformed)
            labels = prediction["labels"].detach().cpu()
            boxes = prediction["boxes"].detach().cpu()
            labels = labels.numpy().tolist()
            boxes = boxes.numpy().tolist()
            data = {'labels': labels, 'boxes': boxes}
            res = jsonify(data)

        except:
            res = jsonify({'error': 'error during prediction'})

    res.headers.set('Access-Control-Allow-Origin', '*')
    res.headers.set('Access-Control-Allow-Methods', 'GET, POST')
    return res


@app.route('/')
def hello():
    return 'Hello World!'
