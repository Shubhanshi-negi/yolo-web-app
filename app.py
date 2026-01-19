from flask import Flask, render_template, request, jsonify
import base64
import numpy as np
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__)

# Load model once
model = YOLO("yolov8n.pt")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.json['image']

        # Decode base64
        encoded = data.split(",")[1]
        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Run YOLO
        results = model(frame, conf=0.45)

        annotated = results[0].plot()

        # Encode output
        _, buffer = cv2.imencode('.jpg', annotated)
        img_str = base64.b64encode(buffer).decode('utf-8')

        # Detected objects
        objects = []
        for box in results[0].boxes:
            cls = int(box.cls[0])
            objects.append(model.names[cls])

        return jsonify({
            "image": img_str,
            "objects": list(set(objects))
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
