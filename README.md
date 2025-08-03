# Facerecognition
Face recognition is a biometric technology that identifies or verifies a person by analyzing and comparing patterns based on their facial features. It uses advanced computer vision and deep learning algorithms to detect faces in images or video and match them with stored face data.
# 🎯 Face Recognition Attendance System

A real-time face recognition-based attendance system using OpenCV, Flask, and optionally the Buffalo model (InsightFace) for robust face detection and recognition.

## 📸 Features

- Real-time face detection and recognition using webcam
- Attendance logging with timestamp
- Face dataset creation
- Preprocessing with face alignment (optional)
- Flask web interface for ease of access
- CSV-based attendance record keeping
- Optionally uses InsightFace (Buffalo) for advanced recognition

## 🧠 Tech Stack

- Python
- OpenCV
- Flask
- InsightFace (optional: RetinaFace + ArcFace)
- NumPy
- Pandas
- SQLite or CSV for storing data

## 📂 Project Structure

face-recognition-attendance/ │ ├── static/                # CSS, JS, images ├── templates/             # HTML files │   ├── index.html │   ├── register.html │   └── attendance.html │ ├── dataset/               # Saved face images ├── embeddings/            # Face embeddings (if using InsightFace) ├── attendance.csv         # Attendance records │ ├── app.py                 # Flask app ├── face_recognition.py    # Core logic (OpenCV or InsightFace) ├── camera.py              # Webcam capture & processing ├── utils.py               # Utility functions (alignment, saving, etc.) └── requirements.txt       # Required packages

## ⚙ Installation

1. *Clone the repository*
   ```bash
   git clone https://github.com/sreelekhakummetha/face-recognition-attendance.git
   cd face-recognition-attendance

2. Create and activate virtual environment (optional)

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


3. Install dependencies

pip install -r requirements.txt


4. Download Pretrained Models (if using InsightFace)

Face Detection: RetinaFace

Face Recognition: ArcFace





🚀 Running the App

python app.py

Open http://127.0.0.1:5000 in your browser.

Use the UI to register a face or take attendance.


✅ How It Works

1. Registration

Capture face images with the user's name.

Save the cropped and aligned faces into dataset/.



2. Training (if using own embeddings)

Extract embeddings for all registered faces and store them.



3. Recognition

During live detection, compare current embedding with stored ones using cosine similarity.

Mark attendance if match is found and not already marked.



4. Attendance

Attendance is saved in attendance.csv with:

Name

Time

Date





📦 Requirements

Python 3.7+

OpenCV

Flask

NumPy

Pandas

insightface (if using Buffalo model)


📄 Example .csv Output

Name,Date,Time
Sreelekha,03/08/2025,10:32:17
Lokeswarreddy,03/08/2025,10:32:45

🛡 License

This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgements

OpenCV

InsightFace by DeepInsight

Flask



---

> Made with ❤ by Sreelekha



---

