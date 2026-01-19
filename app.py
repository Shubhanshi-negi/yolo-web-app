from flask import Flask, render_template, request, jsonify
import base64
import numpy as np
import cv2
from ultralytics import YOLO
import os

app = Flask(__name__)
model = YOLO("yolov8n.pt")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/detect', methods=['POST'])
def detect():
    data = request.json['image']

    # decode base64
    encoded = data.split(",")[1]
    img_bytes = base64.b64decode(encoded)

    # convert to numpy
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = model(img, conf=0.45)
    img = results[0].plot()

    _, buffer = cv2.imencode(".jpg", img)
    img_str = base64.b64encode(buffer).decode()

    return jsonify({"image": img_str})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
