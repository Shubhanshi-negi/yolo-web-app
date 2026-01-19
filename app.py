from flask import Flask, render_template, request, jsonify
import base64
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io

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

    results = model(frame)

    annotated = results[0].plot()
    _, buffer = cv2.imencode('.jpg', annotated)
    img_str = base64.b64encode(buffer).decode('utf-8')

    return jsonify({"image": img_str})

if __name__ == "__main__":
    app.run()
