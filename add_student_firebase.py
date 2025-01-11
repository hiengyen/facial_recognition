import firebase_admin
from firebase_admin import credentials, db

cred = credentials.Certificate("./firebase-adminsdk.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://smart-school-firebase-default-rtdb.asia-southeast1.firebasedatabase.app/"
})

ref_students = db.reference("students")

users = [
    # {"student_id": "CT060409", "student_name": "Tran Luu Dung",
    #  "rfid_code": "1A2B3C4D"},
    {
        "student_id": "Unknown",
        "student_name": "Khong ton tai",
        "rfid_code": "00000000",
    },
    # {
    #     "student_id": "CT060412",
    #     "student_name": "Nguyen Trung Hieu",
    #     "rfid_code": "9FE9721C",
    # },
    # {
    #     "student_id": "CT060406",
    #     "student_name": "Nguyen Minh Duc ",
    #     "rfid_code": "BFA8661F",
    # },
    # {
    #     "student_id": "CT060331",
    #     "student_name": "Dang Minh Phuong",
    #     "rfid_code": "EFF85A1F",
    # },
    #


    # {"student_id": "", "student_name": "","rfid_code": ""}
]

# Using Dictionary
students_data = {}
for user in users:
    student_ref = ref_students.child(user['rfid_code']).set(
        {"student_id": user["student_id"],
         "student_name": user["student_name"],
         "rfid_code": user["rfid_code"]
         })
