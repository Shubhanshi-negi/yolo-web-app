from flask import Flask, render_template, request, jsonify
import base64
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import cv2
import os

app = Flask(__name__)
model = YOLO("yolov8n.pt")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.json.get('image')
        if not data:
            return jsonify({"error": "No image"}), 400

        img_data = base64.b64decode(data.split(',')[1])
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        frame = np.array(img)

        results = model(frame)
        annotated = results[0].plot()

        _, buffer = cv2.imencode('.jpg', annotated)
        img_str = base64.b64encode(buffer).decode('utf-8')

        return jsonify({"image": img_str})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
