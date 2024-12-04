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

# Use NumPy for faster operations
known_face_encodings = np.array(data["encodings"])
known_face_names = data["names"]


# Initialize the USB camera
cap = cv2.VideoCapture(0)

# Initialize variables
cv_scaler = 4  # Hệ số tỉ lệ giảm khung hình (giảm ít hơn để giữ chi tiết)
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
        writer.writerow(["Name", "Date", "Time"])


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
        tolerance = 0.45  # default is 0.6
        matches = face_recognition.compare_faces(
            known_face_encodings, face_encoding, tolerance=tolerance
        )
        name = "Unknown"

        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(
            known_face_encodings, face_encoding
        )
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            # Record recognized person into the CSV file only once
            record_recognized_person_once(name)
            upload_record_to_firebase(name)
        # else:
        #     # Save unknown face to disk for later processing
        #     save_unknown_face(frame, face_locations[i])

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
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler

        # Draw a box around the face
        box_color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        # Green for known, red for unknown
        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 1)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, top - 35),
                      (right, top), box_color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left, top - 6), font, 0.7, (0, 0, 0), 1)

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


def save_unknown_face(frame, face_location):
    # Extract face location
    top, right, bottom, left = face_location
    top *= cv_scaler
    right *= cv_scaler
    bottom *= cv_scaler
    left *= cv_scaler

    # Crop and save the face
    face_image = frame[top:bottom, left:right]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(unknown_faces_dir, f"unknown_{timestamp}.jpg")
    cv2.imwrite(file_path, face_image)
    print(f"[INFO] Saved unknown face to {file_path}")


while True:
    # Capture a frame from USB camera
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture image")
        break

    # Process the frame with the function
    processed_frame = process_frame(frame)

    # Get the text and boxes to be drawn based on the processed frame
    display_frame = draw_results(processed_frame)

    # Calculate and update FPS
    current_fps = calculate_fps()

    # Attach FPS counter to the text and boxes
    cv2.putText(
        display_frame,
        f"FPS: {current_fps:.1f}",
        (display_frame.shape[1] - 150, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    # Display everything over the video feed.
    cv2.imshow("Video", display_frame)

    # Break the loop and stop the script if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
cap.release()
