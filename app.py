import cv2
import time
import numpy as np
import os
from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
from deepface import DeepFace
from flask_cors import CORS
import torch
from dotenv import load_dotenv


app = Flask(__name__)
CORS(app)
load_dotenv() 
# Check if YOLOv8 model exists, otherwise download it
MODEL_PATH = "yolov8s.pt"

if not os.path.exists(MODEL_PATH):
    torch.hub.download_url_to_file("https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt", MODEL_PATH)

# Load the model
model = YOLO(MODEL_PATH)

# model = YOLO("yolov8s.pt")

# Reference object height for scaling (assumed 100 cm)
REFERENCE_OBJECT_HEIGHT_CM = 100

# Harmful objects list
HARMFUL_OBJECTS = ["knife", "gun", "weapon"]

# Folder to store processed images
OUTPUT_FOLDER = "output"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)


def detect_thief(image_path):
    """Processes the image, detects thieves, and estimates details."""
    frame = cv2.imread(image_path)
    if frame is None:
        return {"error": "⚠️ Invalid image file or path!"}

    img_height, img_width, _ = frame.shape
    start_time = time.time()

    # Run YOLO detection
    results = model(frame, conf=0.5)

    thief_detected = False
    detected_objects = []
    harmful_objects_detected = []
    thief_box = None

    # Process detections
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = model.names.get(class_id, "unknown")

            # Ignore low-confidence detections
            if confidence < 0.5:
                continue

            if label == "person":
                thief_detected = True
                thief_box = (x1, y1, x2, y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red Box for Thief
                cv2.putText(frame, "THIEF", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif label in HARMFUL_OBJECTS:
                harmful_objects_detected.append(label)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 3)  # Orange Box for Harmful Objects
                cv2.putText(frame, label.upper(), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                detected_objects.append(label)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green Box for Normal Objects
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    inference_time = round(time.time() - start_time, 2)

    # If no thief is detected, return only detected objects
    if not thief_detected:
        return {
            "thief_detected": False,
            "message": "Thief not detected. Please upload another image as evidence.",
            "detected_objects": detected_objects,
            "harmful_objects": harmful_objects_detected,
            "inference_time": inference_time
        }

    # Initialize age & gender
    age, gender = None, None

    # Perform face analysis if a thief is detected
    try:
        face_analysis = DeepFace.analyze(image_path, actions=["age", "gender"], enforce_detection=False)
        if face_analysis and isinstance(face_analysis, list):
            age = face_analysis[0].get("age", None)
            gender = face_analysis[0].get("dominant_gender", "").capitalize()
            age = max(18, min(age, 80)) if age else None  # Ensure age is realistic
    except Exception as e:
        print(f"⚠️ Face analysis error: {e}")

    # Estimate real-world height
    def estimate_real_height(thief_box, img_height, ref_height_cm):
        """Estimate real-world height of the detected person."""
        if thief_box:
            person_height_pixels = thief_box[3] - thief_box[1]
            scale_factor = ref_height_cm / img_height
            estimated_height_cm = person_height_pixels * scale_factor * 1.2
            return round(max(140, min(estimated_height_cm, 200)), 1)
        return 175 if gender == "Man" else 160

    # Estimate weight based on height & gender
    def estimate_real_weight(height_cm, gender):
        """Estimate weight using an average BMI formula."""
        if height_cm:
            avg_bmi = 23 if gender == "Man" else 21
            return round(avg_bmi * ((height_cm / 100) ** 2), 2)
        return None

    # Compute height & weight
    height_cm = estimate_real_height(thief_box, img_height, REFERENCE_OBJECT_HEIGHT_CM)
    weight_kg = estimate_real_weight(height_cm, gender)

    # Save the processed image
    output_path = os.path.join(OUTPUT_FOLDER, "output.jpg")
    cv2.imwrite(output_path, frame)

    return {
        "thief_detected": True,
        "age": age,
        "gender": gender,
        "estimated_height_cm": height_cm,
        "estimated_weight_kg": weight_kg,
        "detected_objects": detected_objects,
        "harmful_objects": harmful_objects_detected,
        "thief_box": thief_box,
        "inference_time": inference_time,
        "image_path": output_path
    }


@app.route('/detect_thief', methods=['POST'])
def detect_thief_api():
    """API endpoint for thief detection."""
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded!"}), 400

    image = request.files['image']
    image_path = os.path.join(OUTPUT_FOLDER, "input.jpg")
    image.save(image_path)

    # Run detection
    result = detect_thief(image_path)

    return jsonify(result)


@app.route('/get_image', methods=['GET'])
def get_image():
    """API endpoint to fetch the processed image."""
    image_path = os.path.join(OUTPUT_FOLDER, "output.jpg")
    if not os.path.exists(image_path):
        return jsonify({"error": "No processed image available!"}), 404

    return send_file(image_path, mimetype='image/jpeg')


# if __name__ == '__main__':
#     # Run Flask app for Render deployment
#     app.run(host='0.0.0.0', port=10000, debug=True)

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
