# YOLO Indoor Object Detection (Real-Time)

A real-time indoor object detection system built on a pretrained YOLOv8n model trained on COCO classes.  
Optimized for indoor environments and everyday objects.

If the system becomes overly descriptive, feel free to mute it — it will continue working quietly and efficiently.

---

## What This Does

- Uses a pretrained YOLOv8n model (downloads automatically on first run)
- Detects common indoor objects via webcam in real time
- Provides optional audio feedback for accessibility
- Displays FPS, object counts, and confidence levels
- Allows confidence threshold adjustment during execution

---

## Requirements

- Python 3.8 or newer

Install dependencies:

```bash
pip install ultralytics opencv-python pyttsx3
