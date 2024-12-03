import face_recognition
import cv2
import numpy as np
import time
import pickle
import os
import csv
from concurrent.futures import ThreadPoolExecutor

# Load pre-trained face encodings
print("[INFO] Loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = np.array(data["encodings"])
known_face_names = data["names"]

# Initialize the USB camera
cap = cv2.VideoCapture(0)

# Variables
cv_scaler = 4  # Higher value for faster performance
face_locations, face_encodings, face_names = [], [], []
frame_count, start_time, fps = 0, time.time(), 0
unknown_faces_dir, csv_file = "unknown_faces", "recognized_faces.csv"
recognized_names = set()

# Create directories and CSV file if not exist
os.makedirs(unknown_faces_dir, exist_ok=True)
if not os.path.exists(csv_file):
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Date", "Time"])

# Thread pool for parallel tasks
executor = ThreadPoolExecutor(max_workers=2)


def process_frame(frame):
    global face_locations, face_encodings, face_names
    resized_frame = cv2.resize(frame, (0, 0), fx=(1 / cv_scaler), fy=(1 / cv_scaler))
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Detect faces and encode
    face_locations = face_recognition.face_locations(rgb_resized_frame)
    face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations)

    face_names = []
    for i, face_encoding in enumerate(face_encodings):
        tolerance = 0.6
        matches = face_recognition.compare_faces(
            known_face_encodings, face_encoding, tolerance
        )
        distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        name = "Unknown"

        if matches and matches[np.argmin(distances)]:
            name = known_face_names[np.argmin(distances)]
            record_recognized_person_once(name)

        face_names.append(name)
    return frame


def draw_results(frame):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top, right, bottom, left = [x * cv_scaler for x in [top, right, bottom, left]]
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, top - 30), (right, top), color, cv2.FILLED)
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


def calculate_fps():
    global frame_count, start_time, fps
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count, start_time = 0, time.time()
    return fps


def record_recognized_person_once(name):
    if name not in recognized_names:
        recognized_names.add(name)
        timestamp = time.strftime("%H:%M:%S")
        date = time.strftime("%d-%m-%Y")
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([name, date, timestamp])
        print(f"[INFO] Recorded {name} on {date} at {timestamp}")


while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture image")
        break

    # Process every 5th frame to reduce workload
    if frame_count % 5 == 0:
        executor.submit(process_frame, frame)

    # Draw results and calculate FPS
    display_frame = draw_results(frame)
    current_fps = calculate_fps()
    cv2.putText(
        display_frame,
        f"FPS: {current_fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Video", display_frame)
    if cv2.waitKey(1) == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
cap.release()
executor.shutdown()
