import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Giả sử các nhãn thực tế và dự đoán
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]  # Nhãn thực tế
y_pred = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0]  # Nhãn dự đoán

# Tính Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

# Trực quan hóa bằng heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicion')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
