import cv2
import os
from datetime import datetime
import time

# Change this to the name of the person you're photographing
IDENTIFICATION = input("Enter the ID of the student:")
PERSON_NAME = input("Enter the name of the student:")


def create_folder(identification, name):
    dataset_folder = "dataset"
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    # Combine identification and name into the folder name
    folder_name = f"{identification} - {name}"
    person_folder = os.path.join(dataset_folder, folder_name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)
    return person_folder


def capture_photos(identification, name):
    folder = create_folder(identification, name)

    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    # Allow camera to warm up
    time.sleep(2)

    photo_count = 0

    print(f"Taking photos for {name}. Press SPACE to capture, 'q' to quit.")

    while True:
        # Capture frame from USB webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Display the frame
        cv2.imshow("Capture", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):  # Space key
            photo_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{identification}_{name}_{timestamp}.jpg"
            filepath = os.path.join(folder, filename)
            cv2.imwrite(filepath, frame)
            print(f"Photo {photo_count} saved: {filepath}")

        elif key == ord("q"):  # Q key
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print(f"Photo capture completed. {photo_count} photos saved for {name}.")


if __name__ == "__main__":
    capture_photos(IDENTIFICATION, PERSON_NAME)
