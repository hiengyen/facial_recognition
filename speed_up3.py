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
from firebase_admin import credentials, db, auth

# Khởi tạo Firebase Admin SDK
cred = credentials.Certificate(
    "./firebase-adminsdk.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://smart-school-firebase-default-rtdb.asia-southeast1.firebasedatabase.app/"
})


# Load pre-trained face encodings
print("[INFO] Loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = np.array(data["encodings"])
known_face_names = data["names"]

# Initialize the USB camera
cap = cv2.VideoCapture(0)

# Variables
cv_scaler = 4
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


def process_frame(frame):
    global face_locations, face_encodings, face_names
    resized_frame = cv2.resize(frame, (0, 0), fx=(
        1 / cv_scaler), fy=(1 / cv_scaler))
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Detect faces and encode
    face_locations = face_recognition.face_locations(rgb_resized_frame)
    face_encodings = face_recognition.face_encodings(
        rgb_resized_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # Efficient matching with NumPy operations
        distances = np.linalg.norm(
            known_face_encodings - face_encoding, axis=1)
        best_match_index = np.argmin(distances)
        name = "Unknown"

        if distances[best_match_index] < 0.6:  # Tolerance for recognition
            name = known_face_names[best_match_index]
            record_recognized_person_once(name)
            upload_record_to_firebase(name)

        face_names.append(name)
    return frame


def record_recognized_person_once(name):
    # Using vietnam timezone
    vn_tz = pytz.timezone("Asia/Ho_Chi_Minh")
    vn_now = datetime.now(vn_tz)
    current_time = vn_now.strftime("%H:%M:%S")
    current_date = vn_now.strftime("%d-%m-%Y")

    # Only record if the name has not been saved before
    if name not in recognized_names:
        recognized_names.add(name)  # Add to the set

        # Write to the CSV file
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            # Add date and time
            writer.writerow([name, current_date, current_time])
        print(f"[INFO] Recorded {name} on {current_date} at {current_time}")


def upload_record_to_firebase(name):
    vn_tz = pytz.timezone("Asia/Ho_Chi_Minh")
    vn_now = datetime.now(vn_tz)
    current_time = vn_now.strftime("%H:%M:%S")
    current_date = vn_now.strftime("%d-%m-%Y")

    try:
        # Kiểm tra nếu Firebase Admin đã được khởi tạo
        if not firebase_admin._apps:
            raise RuntimeError("Firebase Admin SDK chưa được khởi tạo.")

        # Tham chiếu đến nhánh "recognized_faces"
        ref = db.reference("recognized_faces")
        records = ref.get()  # Lấy tất cả bản ghi từ Firebase

        # Kiểm tra nếu bản ghi đã tồn tại
        for record in records.values() if records else []:
            if record["name"] == name and record["date"] == current_date:
                print(
                    f"[INFO] Record for {name} on {current_date} already exists.")
                return  # Dừng lại nếu bản ghi đã tồn tại

        # Đẩy dữ liệu mới lên Firebase
        ref.push({
            "name": name,
            "date": current_date,
            "time": current_time
        })
        print(
            f"[INFO] Uploaded record for {name} at {current_time} on {current_date} to Firebase")
    except RuntimeError as re:
        print(f"[ERROR] Initialization error: {re}")
    except Exception as e:
        print(f"[ERROR] Failed to upload record for {name} to Firebase: {e}")


def draw_results(frame):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top, right, bottom, left = [
            x * cv_scaler for x in [top, right, bottom, left]]
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

    # Frame skipping: Process every 5th frame
    if frame_count % 5 == 0:
        process_frame(frame)

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
