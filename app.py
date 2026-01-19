from flask import Flask, render_template, request, jsonify
import base64
import numpy as np
from ultralytics import YOLO
import cv2
from PIL import Image
import io
import os

app = Flask(__name__)
model = YOLO("yolov8n.pt")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/detect', methods=['POST'])
def detect():
    data = request.json['image']
    img_data = base64.b64decode(data.split(',')[1])

    img = Image.open(io.BytesIO(img_data))
    frame = np.array(img)

    results = model(frame, conf=0.45)

    annotated = results[0].plot()

    _, buffer = cv2.imencode('.jpg', annotated)
    img_str = base64.b64encode(buffer).decode('utf-8')

    objects = []
    for box in results[0].boxes:
        cls = int(box.cls[0])
        objects.append(model.names[cls])

    return jsonify({
        "image": img_str,
        "objects": list(set(objects))
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
