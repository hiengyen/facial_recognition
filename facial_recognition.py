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
import threading


# Init  Firebase Admin SDK
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
# List contain "id" and "name"
known_student_info = data["student_info"]
# Initialize the USB camera
cap = cv2.VideoCapture(0)

# Initialize variables
cv_scaler = 2
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


def process_frame(frame):
    global face_locations, face_encodings, face_names

    # Resize the frame using cv_scaler to increase performance
    resized_frame = cv2.resize(frame, (0, 0), fx=(
        1 / cv_scaler), fy=(1 / cv_scaler))

    # Convert the image from BGR to RGB colour space
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_resized_frame)
    face_encodings = face_recognition.face_encodings(
        rgb_resized_frame, face_locations)

    face_names = []
    for i, face_encoding in enumerate(face_encodings):
        # Adjust tolerance for comparison (lower tolerance = stricter matching)
        tolerance = 0.45
        matches = face_recognition.compare_faces(
            known_face_encodings, face_encoding, tolerance=tolerance
        )
        name = "Unknown"
        student_id = "Unknown"

        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(
            known_face_encodings, face_encoding
        )
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            matched_info = known_student_info[best_match_index]
            student_id = matched_info["id"]
            student_name = matched_info["name"]
            name = f"{student_id} - {student_name}"
            async_upload(student_id, student_name)
            async_record(student_id, student_name)
        else:
            # Save unknown face for later processing
            save_unknown_face(frame, face_locations[i])

        face_names.append(name)

    return frame


def async_upload(student_id, student_name):
    threading.Thread(
        target=upload_record_to_firebase, args=(student_id, student_name)
    ).start()


def async_record(student_id, student_name):
    threading.Thread(
        target=record_recognized_person_once, args=(student_id, student_name)
    ).start()


def record_recognized_person_once(student_id, student_name):
    # Using Vietnam timezone
    vn_tz = pytz.timezone("Pacific/Kiritimati")
    # vn_tz = pytz.timezone("Asia/Ho_Chi_Minh")
    vn_now = datetime.now(vn_tz)
    current_time = vn_now.strftime("%H:%M:%S")
    current_date = vn_now.strftime("%d-%m-%Y")

    if student_id not in recognized_names:
        recognized_names.add(student_id)
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([student_id, student_name,
                            current_date, current_time])
        print(
            f"[INFO] Recorded {student_id} - {student_name} on {current_date} at {current_time}")
        return


def upload_record_to_firebase(student_id, student_name):
    # vn_tz = pytz.timezone("Pacific/Kiritimati")
    vn_tz = pytz.timezone("Asia/Ho_Chi_Minh")
    vn_now = datetime.now(vn_tz)
    current_time = vn_now.strftime("%H:%M:%S")
    current_date = vn_now.strftime("%d-%m-%Y")

    ref_students = db.reference("students")

    users = [
        {
            "student_id": "CT060412",
            "student_name": "Nguyen Trung Hieu",
            "rfid_code": "9FE9721C",
        },
        {
            "student_id": "CT060406",
            "student_name": "Nguyen Minh Duc ",
            "rfid_code": "BFA8661F",
        },
        {
            "student_id": "CT060331",
            "student_name": "Nguyen Minh Phuong",
            "rfid_code": "EFF85A1F",
        },
    ]

    if not ref_students.get():  # if branch not exists
        for user in users:
            ref_students.child(user["rfid_code"]).set(
                {
                    "student_id": user["student_id"],
                    "student_name": user["student_name"],
                    "rfid_code": user["rfid_code"],
                }
            )

    try:
        ref_face_record = db.reference("recognized_faces")
        ref_attendance_record = db.reference("students_attendance")
        # get all record
        records_today = ref_face_record.child(
            current_date).child(student_id).get()

        if records_today:
            print(
                f"[INFO] Record for student_id '{student_id}' on date '{current_date}' already exists."
            )
            return

        ref_face_record.child(current_date).child(student_id).set(
            {
                "student_name": student_name,
                "date": current_date,
                "time": current_time,
                "state": 0,
            }
        )
        ref_attendance_record.child(current_date).child(student_id).set(
            {
                "student_name": student_name,
                "date": current_date,
                "checkin": current_time,
                "checkout": "",
                "state": "",
            }
        )

        # print(
        #     f"[INFO] Uploaded record for {student_name} at {current_time} on {current_date} to Firebase")
    except RuntimeError as re:
        print(f"[ERROR] Initialization error: {re}")
    except Exception as e:
        print(
            f"[ERROR] Failed to upload record for {student_name} to Firebase: {e}"
        )


def draw_results(frame):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler

        # Draw a box around the face
        box_color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)

        # Draw a label with the name below the face
        cv2.rectangle(frame, (left, top - 35),
                      (right, top), box_color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, top - 6), font, 0.5, (0, 0, 0), 1)

    return frame


def save_unknown_face(frame, face_location):
    top, right, bottom, left = face_location
    top *= cv_scaler
    right *= cv_scaler
    bottom *= cv_scaler
    left *= cv_scaler

    face_image = frame[top:bottom, left:right]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(unknown_faces_dir, f"unknown_{timestamp}.jpg")
    cv2.imwrite(file_path, face_image)
    print(f"[INFO] Saved unknown face to {file_path}")


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

    # processed_frame = process_frame(frame)
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

    cv2.namedWindow("Video")
    cv2.moveWindow("Video", 900, 0)
    cv2.imshow("Video", display_frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
