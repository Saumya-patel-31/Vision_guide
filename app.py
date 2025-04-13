from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)

model = YOLO("yolov8n.pt")

custom_map = {
    "tv": "door",
    "refrigerator": "door",
    "bench": "stairs",
    "ladder": "stairs"
}

def describe_with_direction(label, x_center, width):
    if x_center < width * 0.33:
        return f"{label} to your left"
    elif x_center > width * 0.66:
        return f"{label} to your right"
    else:
        return f"{label} ahead"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    frame = np.array(image)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    height, width = frame.shape[:2]

    results = model(frame_rgb, imgsz=320, verbose=False)
    names = model.names
    descriptions = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = names[cls]
            label = custom_map.get(label, label)
            x_center = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
            descriptions.append(describe_with_direction(label, x_center, width))

    text = pytesseract.image_to_string(frame_rgb)
    cleaned_text = text.strip().replace('\n', ' ')
    full_description = ". ".join(set(descriptions))
    if cleaned_text:
        full_description += f". Sign says: {cleaned_text}"

    return jsonify({"description": full_description.strip()})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
