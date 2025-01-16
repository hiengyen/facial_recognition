import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Load pre-trained face encodings
print("[INFO] Loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())

# Use NumPy for faster operations
known_face_encodings = np.array(data["encodings"])
# List contain "id" and "name"
known_student_info = data["student_info"]

# Dữ liệu nhãn thực tế (y_true)
y_true = [info['id'] for info in known_student_info]

# Giả sử nhãn dự đoán (y_pred) được tạo ra từ mô hình
# Ở đây, giả sử mô hình dự đoán đúng tất cả các nhãn (để minh họa)
y_pred = y_true  # Thay thế bằng kết quả dự đoán thực tế từ mô hình của bạn

# Danh sách các lớp (mã sinh viên)
classes = ["CT060412", "CT060406", "CT060331", "background"]

# Tính Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=classes)

# Trực quan hóa bằng heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Tính các chỉ số đánh giá
accuracy = accuracy_score(y_true, y_pred)
# Sử dụng 'weighted' cho đa lớp
precision = precision_score(y_true, y_pred, average='weighted')
# Sử dụng 'weighted' cho đa lớp
recall = recall_score(y_true, y_pred, average='weighted')
# Sử dụng 'weighted' cho đa lớp
f1 = f1_score(y_true, y_pred, average='weighted')

# Tính Specificity và Negative Predictive Value (NPV)
# Specificity và NPV cần được tính riêng cho từng lớp trong bài toán đa lớp
specificity_list = []
npv_list = []

for i, cls in enumerate(classes):
    # Loại bỏ hàng và cột của lớp hiện tại
    TN = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
    FP = np.sum(cm[:, i]) - cm[i, i]  # Tổng cột trừ đi TP
    FN = np.sum(cm[i, :]) - cm[i, i]  # Tổng hàng trừ đi TP
    TP = cm[i, i]

    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    npv = TN / (TN + FN) if (TN + FN) != 0 else 0

    specificity_list.append(specificity)
    npv_list.append(npv)

# Tính giá trị trung bình của Specificity và NPV
specificity_avg = np.mean(specificity_list)
npv_avg = np.mean(npv_list)

# In kết quả
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Specificity (Average): {specificity_avg:.4f}")
print(f"Negative Predictive Value (Average): {npv_avg:.4f}")
print(f"F1 Score: {f1:.4f}")
