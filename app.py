from flask import Flask, render_template, Response, jsonify
import cv2
import face_recognition
import dlib
import numpy as np
import os
from datetime import datetime
import pandas as pd
import time

app = Flask(__name__)

# Load known faces
known_face_encodings = []
known_face_names = []

def load_known_faces():
    known_face_encodings.clear()
    known_face_names.clear()
    for name in os.listdir('known_faces'):
        for filename in os.listdir(f'known_faces/{name}'):
            image = face_recognition.load_image_file(f'known_faces/{name}/{filename}')
            encoding = face_recognition.face_encodings(image)
            if len(encoding) > 0:
                known_face_encodings.append(encoding[0])
                known_face_names.append(name)

# EAR function
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Global camera stream
camera = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('index.html')

# Stream video to browser
def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Detection logic (runs on Detect button click)
@app.route('/detect', methods=['POST'])
def detect():
    load_known_faces()
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    EAR_THRESHOLD = 0.2
    EAR_CONSEC_FRAMES = 1
    FRAME_DURATION = 5  # seconds

    start_time = time.time()
    blink_counter = {}
    total_blinks = {}
    attendance = {}

    today_str = datetime.now().strftime("%Y-%m-%d")
    attendance_file = "attendance.csv"

    if os.path.exists(attendance_file):
        existing_df = pd.read_csv(attendance_file)
        existing_records = set(zip(existing_df["Name"], existing_df["Date"]))
    else:
        existing_df = pd.DataFrame(columns=["Name", "Date", "Timestamp"])
        existing_records = set()

    while time.time() - start_time < FRAME_DURATION:
        success, frame = camera.read()
        if not success:
            continue

        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = face_detector(gray, 0)

        for (face_encoding, face_location, rect) in zip(face_encodings, face_locations, rects):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

                shape = landmark_predictor(gray, rect)
                shape_np = np.zeros((68, 2), dtype=int)
                for i in range(68):
                    shape_np[i] = (shape.part(i).x, shape.part(i).y)

                left_eye = shape_np[42:48]
                right_eye = shape_np[36:42]
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0

                if name not in blink_counter:
                    blink_counter[name] = 0
                    total_blinks[name] = 0

                if ear < EAR_THRESHOLD:
                    blink_counter[name] += 1
                else:
                    if blink_counter[name] >= EAR_CONSEC_FRAMES:
                        total_blinks[name] += 1
                        blink_counter[name] = 0

    results = []
    for name, blinks in total_blinks.items():
        attendance_key = (name, today_str)
        if attendance_key in attendance or attendance_key in existing_records:
            results.append(f"{name}: Already marked today")
        elif blinks > 0:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            attendance[attendance_key] = timestamp
            results.append(f"{name}: Attendance marked")
        else:
            results.append(f"{name}: No blink detected")

    if attendance:
        new_df = pd.DataFrame(
            [(name, date, timestamp) for ((name, date), timestamp) in attendance.items()],
            columns=["Name", "Date", "Timestamp"]
        )
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        updated_df.to_csv(attendance_file, index=False)

    return jsonify({"status": "success", "message": "\n".join(results)})


if __name__ == '__main__':
    app.run(debug=True)
