import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, Response, jsonify
import time
import csv
from datetime import datetime

# ----------------- CONFIGURATION -------------------
DESIRED_FPS = 10  # Set desired FPS (lower = less CPU usage)
FRAME_INTERVAL = int(30 / DESIRED_FPS)
SIMILARITY_THRESHOLD = 0.4  # Set similarity threshold to avoid false positives
# ---------------------------------------------------

# Load saved student embeddings
student_embeddings = np.load('student_embeddings.npy', allow_pickle=True).item()

# Initialize the InsightFace model
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

# Flask app
app_flask = Flask(__name__)

# Camera settings
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

attendance_list = []
current_student = {"name": "Not Identified", "roll_number": "", "status": "Unknown"}
last_positions = {}

# Direction detection
def get_direction(face):
    x = int(face.bbox[0])
    track_id = face.track_id if hasattr(face, 'track_id') else 0

    if track_id not in last_positions:
        last_positions[track_id] = x
        return None

    old_x = last_positions[track_id]
    last_positions[track_id] = x

    if x - old_x > 20:
        return "entering"
    elif x - old_x < -20:
        return "exiting"
    else:
        return None

# CSV logging
def log_attendance(name, roll_number, action):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open('attendance_log.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, roll_number, action, timestamp])

# Video feed generator
def generate_frames():
    global current_student
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % FRAME_INTERVAL != 0:
            continue  # Skip frames based on desired FPS

        time.sleep(1 / DESIRED_FPS)  # Frame delay to match FPS

        faces = app.get(frame)

        if faces:
            face = faces[0]
            face_embedding = face.embedding
            direction = get_direction(face)

            similarities = {
                name: cosine_similarity([face_embedding], [embedding])[0][0]
                for name, embedding in student_embeddings.items()
            }

            if similarities:
                predicted_name = max(similarities, key=similarities.get)
                max_similarity = similarities[predicted_name]

                if max_similarity >= SIMILARITY_THRESHOLD:
                    name_parts = predicted_name.split('_')
                    name = " ".join(name_parts[:-1])
                    roll_number = name_parts[-1]

                    if direction == "entering":
                        if not any(student['roll_number'] == roll_number for student in attendance_list):
                            attendance_list.append({
                                "name": name,
                                "roll_number": roll_number,
                                "status": "Present"
                            })
                            current_student.update({
                                "name": name,
                                "roll_number": roll_number,
                                "status": "Present"
                            })
                            log_attendance(name, roll_number, "Entry")

                    elif direction == "exiting":
                        current_student.update({
                            "name": name,
                            "roll_number": roll_number,
                            "status": "Exit"
                        })
                        log_attendance(name, roll_number, "Exit")

                    cv2.putText(frame, f"Student: {name}", (200, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"Roll: {roll_number}", (200, 120), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 0, 255), 2, cv2.LINE_AA)
                else:
                    current_student = {"name": "Not Identified", "roll_number": "", "status": "Unknown"}
                    cv2.putText(frame, "Unknown Face", (250, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                current_student = {"name": "Not Identified", "roll_number": "", "status": "Unknown"}
                cv2.putText(frame, "No face detected", (300, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            current_student = {"name": "Not Identified", "roll_number": "", "status": "Unknown"}
            cv2.putText(frame, "No face detected", (300, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Flask routes
@app_flask.route('/student_info')
def student_info():
    return jsonify(current_student)

@app_flask.route('/')
def index():
    return render_template('index.html', student=current_student)

@app_flask.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Initialize CSV if not present
if __name__ == "__main__":
    try:
        with open('attendance_log.csv', 'x', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Roll Number", "Action", "Timestamp"])
    except FileExistsError:
        pass

    app_flask.run(debug=True)
