import sqlite3
import cv2
from datetime import datetime
from tensorflow.keras.models import load_model
import numpy as np

# Load model CNN wajah
model = load_model('model/face_recognition_cnn.h5')
labels = ['rehan', 'engkay']  # sesuaikan label dengan hasil training

# Load Haar Cascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Koneksi ke SQLite
# conn = sqlite3.connect('database/absensi.db')
# cursor = conn.cursor()

# Buka kamera
cam = cv2.VideoCapture(0)

print("Mulai absensi... Tekan 'q' untuk keluar.")

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (150, 150))  # ukuran sesuai model
        face_array = face_img / 255.0
        face_array = np.expand_dims(face_array, axis=0)

        pred = model.predict(face_array)[0]
        label_idx = np.argmax(pred)
        confidence = pred[label_idx]

        if confidence > 0.8:  # ambang kepercayaan
            name = labels[label_idx]
            waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # cursor.execute("INSERT INTO absensi (nama, waktu) VALUES (?, ?)", (name, waktu))
            # conn.commit()
            print(f"Absensi {name} tercatat pada {waktu}")
        else:
            name = "Tidak Dikenal"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

    cv2.imshow("Absensi Wajah", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
conn.close()
