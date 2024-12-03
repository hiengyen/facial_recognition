import os
from imutils import paths
import face_recognition
import pickle
import cv2


print("[INFO] Start processing faces...")
imagePaths = list(paths.list_images("dataset"))
knownEncodings = []
knownStudentInfo = []  # Danh sách chứa thông tin sinh viên (mã + tên)

for i, imagePath in enumerate(imagePaths):
    print(f"[INFO] Processing image {i + 1}/{len(imagePaths)}")

    # Lấy tên thư mục (mã sinh viên - tên)
    folder_name = imagePath.split(os.path.sep)[-2]

    try:
        # Tách mã sinh viên và tên
        student_id, student_name = folder_name.split(" - ", 1)
    except ValueError:
        print(
            f"[WARNING] Folder '{folder_name}' không đúng định dạng 'mã sinh viên - tên'. Bỏ qua...")
        continue

    # Đọc ảnh và xử lý
    image = cv2.imread(imagePath)
    scale_factor = 0.5  # Tỉ lệ giảm kích thước ảnh
    resized_image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
    rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    # Phát hiện và mã hóa khuôn mặt
    boxes = face_recognition.face_locations(rgb, model="cnn")
    encodings = face_recognition.face_encodings(rgb, boxes)

    for encoding in encodings:
        knownEncodings.append(encoding)
        knownStudentInfo.append({"id": student_id, "name": student_name})

# Lưu dữ liệu đã mã hóa
print("[INFO] Serializing encodings...")
data = {"encodings": knownEncodings, "student_info": knownStudentInfo}
with open("encodings.pickle", "wb") as f:
    f.write(pickle.dumps(data))

print("[INFO] Training complete. Encodings saved to 'encodings.pickle'")
