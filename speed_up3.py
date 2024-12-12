import face_recognition
import cv2
import numpy as np
import time
import pickle
import os
import csv
import pytz
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db
from concurrent.futures import ThreadPoolExecutor

# Init Firebase Admin SDK
cred = credentials.Certificate("./firebase-adminsdk.json")
firebase_admin.initialize_app(
    cred,
    {
        "databaseURL": "https://smart-school-firebase-default-rtdb.asia-southeast1.firebasedatabase.app/"
    },
)

# Load pre-trained face encodings
print("[INFO] Loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())

# Use NumPy for faster operations
known_face_encodings = np.array(data["encodings"])
known_student_info = data["student_info"]  # List contains "id" and "name"

# Initialize the USB camera
cap = cv2.VideoCapture(0)

# Initialize variables
cv_scaler = 4
face_locations = []
face_encodings = []
face_names = []
frame_count = 0
start_time = time.time()
fps = 0
unknown_faces_dir = "unknown_faces"  # Unknown directory
csv_file = "recognized_faces.csv"  # CSV file

# Track recognized names to avoid duplicate CSV entries
recognized_names = set()

# Create the directory for unknown faces if it doesn't exist
if not os.path.exists(unknown_faces_dir):
    os.makedirs(unknown_faces_dir)

# Create or open the CSV file for recording recognized faces
if not os.path.exists(csv_file):
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["StudentID", "Name", "Date", "Time"])

# ThreadPoolExecutor for concurrent processing
executor = ThreadPoolExecutor(max_workers=4)


def process_face(face_encoding, i, frame):
    tolerance = 0.45
    matches = face_recognition.compare_faces(
        known_face_encodings, face_encoding, tolerance=tolerance
    )
    name = "Unknown"
    student_id = "Unknown"

    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)

    if matches[best_match_index]:
        matched_info = known_student_info[best_match_index]
        student_id = matched_info["id"]
        student_name = matched_info["name"]
        name = f"{student_id} - {student_name}"
        async_upload(student_id, student_name)
        async_record(student_id, student_name)
    return name


def process_frame(frame):
    global face_locations, face_encodings, face_names

    resized_frame = cv2.resize(frame, (0, 0), fx=(1 / cv_scaler), fy=(1 / cv_scaler))
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_resized_frame)
    face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations)

    face_names.clear()
    tasks = [
        executor.submit(process_face, face_encodings[i], i, frame)
        for i in range(len(face_encodings))
    ]
    for task in tasks:
        face_names.append(task.result())
    return frame


def async_upload(student_id, student_name):
    executor.submit(upload_record_to_firebase, student_id, student_name)


def async_record(student_id, student_name):
    executor.submit(record_recognized_person_once, student_id, student_name)


def record_recognized_person_once(student_id, student_name):
    vn_tz = pytz.timezone("Asia/Ho_Chi_Minh")
    vn_now = datetime.now(vn_tz)
    current_time = vn_now.strftime("%H:%M:%S")
    current_date = vn_now.strftime("%d-%m-%Y")

    if student_id not in recognized_names:
        recognized_names.add(student_id)
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([student_id, student_name, current_date, current_time])
        print(
            f"[INFO] Recorded {
              student_id} - {student_name} on {current_date} at {current_time}"
        )


def upload_record_to_firebase(student_id, student_name):
    vn_tz = pytz.timezone("Asia/Ho_Chi_Minh")
    vn_now = datetime.now(vn_tz)
    current_time = vn_now.strftime("%H:%M:%S")
    current_date = vn_now.strftime("%d-%m-%Y")

    ref_students = db.reference("students")
    if not ref_students.get():
        users = [
            {
                "student_id": "CT060412",
                "student_name": "Nguyen Trung Hieu",
                "rfid_code": "9FE9721C",
            },
            {
                "student_id": "CT060406",
                "student_name": "Nguyen Minh Duc",
                "rfid_code": "BFA8661F",
            },
            {
                "student_id": "CT060331",
                "student_name": "Nguyen Minh Phuong",
                "rfid_code": "EFF85A1F",
            },
        ]
        for user in users:
            ref_students.child(user["rfid_code"]).set(user)

    try:
        ref_face_record = db.reference("recognized_faces")
        ref_attendance_record = db.reference("students_attendance")

        if ref_face_record.child(student_id).get():
            return

        ref_face_record.child(student_id).set(
            {
                "student_name": student_name,
                "date": current_date,
                "time": current_time,
                "state": 0,
            }
        )
        ref_attendance_record.child(student_id).set(
            {
                "student_name": student_name,
                "date": current_date,
                "checkin": current_time,
                "checkout": "",
                "state": "",
            }
        )
    except Exception as e:
        print(
            f"[ERROR] Failed to upload record for {
              student_name} to Firebase: {e}"
        )


def draw_results(frame):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler

        box_color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
        cv2.rectangle(frame, (left, top - 35), (right, top), box_color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, top - 6), font, 0.5, (0, 0, 0), 1)
    return frame


def calculate_fps():
    global frame_count, start_time, fps
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    return fps


while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture image")
        break
    if frame_count % 10 == 0:
        processed_frame = process_frame(frame)

    display_frame = draw_results(processed_frame)
    current_fps = calculate_fps()
    cv2.putText(
        display_frame,
        f"FPS: {current_fps:.1f}",
        (display_frame.shape[1] - 150, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Video", display_frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
