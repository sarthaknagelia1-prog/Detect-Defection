import cv2
import threading
import numpy as np
from ultralytics import YOLO
from flask import Flask, jsonify, render_template, request, redirect, url_for, session, Response
from collections import deque
import serial
import time

app = Flask(__name__)
app.secret_key = "techfuture_secret_key"

# -----------------------------
# SERIAL CONNECTION
# -----------------------------
arduino = serial.Serial("COM14", 9600)
time.sleep(2)

# -----------------------------
# Login
# -----------------------------
USERNAME = "admin"
PASSWORD = "1234"

# -----------------------------
# Load YOLO Model
# -----------------------------
model = YOLO("best.pt")

# -----------------------------
# Global Variables
# -----------------------------
total_products = 0
correct_products = 0
defect_products = 0
stable_class = "No Chalk"
avg_conf = 0.0

confidence_threshold = 0.85
prediction_window = deque(maxlen=15)
confidence_window = deque(maxlen=15)
processing_object = False

defect_classes = [
    "bending_flexural_crack",
    "tensile_mode_I_crack",
    "torsional_helical_crack"
]

lock = threading.Lock()

camera = cv2.VideoCapture(1)

# =====================================================
# DETECTION THREAD
# =====================================================

def detect():
    global total_products, correct_products, defect_products
    global stable_class, avg_conf, processing_object

    while True:
        ret, frame = camera.read()
        if not ret:
            continue

        h, w, _ = frame.shape
        box_size = 300
        x1 = w // 2 - box_size // 2
        y1 = h // 2 - box_size // 2
        x2 = x1 + box_size
        y2 = y1 + box_size

        roi = frame[y1:y2, x1:x2]

        if arduino.in_waiting > 0:
            message = arduino.readline().decode().strip()
            if message == "DETECTED":
                processing_object = True
                prediction_window.clear()
                confidence_window.clear()

        if processing_object:
            results = model.predict(roi, imgsz=224, verbose=False)
            probs = results[0].probs
            class_id = probs.top1
            class_name = model.names[class_id]
            confidence = probs.top1conf.item()

            if confidence > confidence_threshold:
                prediction_window.append(class_name)
                confidence_window.append(confidence)

            if len(prediction_window) == prediction_window.maxlen:
                stable = max(set(prediction_window), key=prediction_window.count)
                avg = float(np.mean(confidence_window))

                with lock:
                    total_products += 1
                    stable_class = stable
                    avg_conf = avg

                    if stable in defect_classes:
                        defect_products += 1
                        arduino.write(b"DEFECT\n")
                    elif stable == "no_crack":
                        correct_products += 1
                        arduino.write(b"GOOD\n")

                processing_object = False
                prediction_window.clear()
                confidence_window.clear()

# =====================================================
# VIDEO STREAM
# =====================================================

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# =====================================================
# ROUTES
# =====================================================

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form["username"] == USERNAME and request.form["password"] == PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("dashboard"))
        return render_template("login.html", error="Invalid Credentials")
    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return render_template("index.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/data")
def data():
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 401

    with lock:
        defect_percentage = (defect_products / total_products * 100) if total_products > 0 else 0

        return jsonify({
            "total": total_products,
            "correct": correct_products,
            "defective": defect_products,
            "defect_percentage": round(defect_percentage, 2),
            "current_class": stable_class,
            "confidence": round(avg_conf, 2)
        })

# =====================================================

if __name__ == "__main__":
    threading.Thread(target=detect, daemon=True).start()
    app.run(debug=False, use_reloader=False)