import firebase_admin
from firebase_admin import credentials, db

# Khởi tạo Firebase Admin SDK
cred = credentials.Certificate("./firebase-adminsdk.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://smart-school-firebase-default-rtdb.asia-southeast1.firebasedatabase.app/"
})

# Tham chiếu tới nhánh "students"
ref_students = db.reference("students")

# Danh sách thông tin 5 người dùng mặc định
users = [
    {"student_id": "CT060409", "student_name": "Tran Luu Dung",
     "rfid_code": "1A2B3C4D"},

    # {"student_id": "", "student_name": "","rfid_code": ""}
]

# Tạo một từ điển chứa tất cả các bản ghi sinh viên
students_data = {}
for user in users:
    student_ref = ref_students.child(user['rfid_code']).set(
        {"student_id": user["student_id"],
         "student_name": user["student_name"],
         "rfid_code": user["rfid_code"]
         })
