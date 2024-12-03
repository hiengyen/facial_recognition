import face_recognition
import cv2
import numpy as np
import time
import pickle
import os
import csv
from multiprocessing.pool import ThreadPool

# Load pre-trained face encodings
print("[INFO] Loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
# Ensure it's a NumPy array for speed
known_face_encodings = np.array(data["encodings"])
known_face_names = data["names"]

# Initialize USB camera
cap = cv2.VideoCapture(0)

# Initialize variables
cv_scaler = 2  # Scale down for faster processing
face_locations, face_encodings, face_names = [], [], []
frame_count, start_time, fps = 0, time.time(), 0
unknown_faces_dir = "unknown_faces"
csv_file = "recognized_faces.csv"
recognized_names = set()

# Create directories and CSV file if not exist
os.makedirs(unknown_faces_dir, exist_ok=True)
if not os.path.exists(csv_file):
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Date", "Time"])


def process_frame(frame):
    global face_locations, face_encodings, face_names
    # Resize frame
    resized_frame = cv2.resize(frame, (0, 0), fx=(1 / cv_scaler), fy=(1 / cv_scaler))
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Detect faces and encodings
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    if not face_locations:
        return frame  # Skip if no faces detected

    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Match faces
    face_names = []
    for i, encoding in enumerate(face_encodings):
        distances = face_recognition.face_distance(known_face_encodings, encoding)
        best_match_idx = np.argmin(distances)
        name = (
            known_face_names[best_match_idx]
            if distances[best_match_idx] < 0.5
            else "Unknown"
        )

        if name == "Unknown":
            save_unknown_face(frame, face_locations[i])
        else:
            record_recognized_person_once(name)

        face_names.append(name)

    return frame


def draw_results(frame):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale up coordinates
        top, right, bottom, left = (
            top * cv_scaler,
            right * cv_scaler,
            bottom * cv_scaler,
            left * cv_scaler,
        )
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        # Draw rectangle and label
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, top - 35), (right, top), color, cv2.FILLED)
        cv2.putText(
            frame,
            name,
            (left + 6, top - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
    return frame


def save_unknown_face(frame, face_location):
    top, right, bottom, left = face_location
    top, right, bottom, left = (
        top * cv_scaler,
        right * cv_scaler,
        bottom * cv_scaler,
        left * cv_scaler,
    )
    face_image = frame[top:bottom, left:right]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(os.path.join(unknown_faces_dir, f"unknown_{timestamp}.jpg"), face_image)


def record_recognized_person_once(name):
    if name not in recognized_names:
        recognized_names.add(name)
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [name, time.strftime("%d-%m-%Y"), time.strftime("%H:%M:%S")]
            )


def calculate_fps():
    global frame_count, start_time, fps
    frame_count += 1
    elapsed = time.time() - start_time
    if elapsed > 1:
        fps = frame_count / elapsed
        frame_count = 0
        start_time = time.time()
    return fps


# Thread pool for parallelizing frame processing
pool = ThreadPool(processes=1)
async_result = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame asynchronously
    if async_result is None or async_result.ready():
        async_result = pool.apply_async(process_frame, (frame,))
    else:
        frame = draw_results(frame)

    # Draw FPS
    cv2.putText(
        frame,
        f"FPS: {calculate_fps():.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
