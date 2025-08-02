import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# Load saved student embeddings
student_embeddings = np.load('student_embeddings.npy', allow_pickle=True).item()

# Initialize the InsightFace model and force it to use GPU (or CPU if needed)
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # Use CPU if you have no GPU
app.prepare(ctx_id=0)  # Set ctx_id to 0 for CPU (1 for GPU)

# Use the laptop camera (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Set the capture resolution to HD (1920x1080)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Set width to 1920 pixels (HD resolution)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Set height to 1080 pixels (HD resolution)

# Get screen resolution (optional, in case you want to adjust the frame size)
screen_width = 1920  # Set the screen width (HD)
screen_height = 1080  # Set the screen height (HD)

# Open the OpenCV window in full-screen mode
cv2.namedWindow("Live Camera", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Live Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    # Capture a frame from the webcam stream
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Resize the frame to fit the screen resolution (you can skip resizing if the frame is already HD)
    frame_resized = cv2.resize(frame, (screen_width, screen_height))

    # Get faces from the frame
    faces = app.get(frame_resized)

    if faces:
        # Use the first detected face
        face_embedding = faces[0].embedding

        # Compare the current face with the saved embeddings using cosine similarity
        similarities = {name: cosine_similarity([face_embedding], [embedding])[0][0]
                        for name, embedding in student_embeddings.items()}

        if similarities:
            # Find the student with the highest similarity
            predicted_name = max(similarities, key=similarities.get)
        else:
            predicted_name = "Unknown"

        # Split the name and roll number
        name_parts = predicted_name.split('_')
        
        # Join the name parts (everything except the last part)
        name = " ".join(name_parts[:-1])

        # The roll number is the last part
        roll_number = name_parts[-1]

        # Define bright and bold colors (Cyan and Magenta in BGR format)
        name_color = (255, 255, 0)  # Cyan (bright and bold)
        roll_number_color = (255, 0, 255)  # Magenta (bold and vibrant)

        # Display the student's name and roll number vertically with bold text
        cv2.putText(frame_resized, f"Student: {name}", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, name_color, 4, lineType=cv2.LINE_AA)  # Increase thickness
        cv2.putText(frame_resized, f"Roll: {roll_number}", 
                    (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, roll_number_color, 4, lineType=cv2.LINE_AA)  # Increase thickness

    else:
        # Display a message if no face is detected
        cv2.putText(frame_resized, "No face detected", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, lineType=cv2.LINE_AA)

    # Display the live video feed in full-screen
    cv2.imshow("Live Camera", frame_resized)

    # Exit the webcam feed if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
