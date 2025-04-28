import sqlite3
import cv2
import face_recognition
import json
from datetime import datetime

# Koneksi ke SQLite
conn = sqlite3.connect('database/absensi.db')
cursor = conn.cursor()

# Load data wajah
cursor.execute("SELECT nama, encoding FROM faces")
data_wajah = cursor.fetchall()

known_encodings = []
known_names = []

for nama, encoding_json in data_wajah:
    encoding = json.loads(encoding_json)
    known_encodings.append(encoding)
    known_names.append(nama)

# Buka kamera
cam = cv2.VideoCapture(0)

print("Mulai absensi... Tekan 'q' untuk keluar.")

while True:
    ret, frame = cam.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Tidak Dikenal"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]

            # Simpan ke database absensi
            waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute("INSERT INTO absensi (nama, waktu) VALUES (?, ?)", (name, waktu))
            conn.commit()
            print(f"Absensi {name} tercatat pada {waktu}")

        # Gambar kotak dan nama
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

    cv2.imshow("Absensi Wajah", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
conn.close()
