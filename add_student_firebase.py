import firebase_admin
from firebase_admin import credentials, db

cred = credentials.Certificate("./firebase-adminsdk.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://smart-school-firebase-default-rtdb.asia-southeast1.firebasedatabase.app/"
})

ref_students = db.reference("students")

users = [
    {"student_id": "CT060409", "student_name": "Tran Luu Dung",
     "rfid_code": "1A2B3C4D"},

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
