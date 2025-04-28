import sqlite3
import cv2
import face_recognition
import json
import os

# Buat folder jika belum ada
if not os.path.exists('database'):
    os.makedirs('database')

# Koneksi ke SQLite
conn = sqlite3.connect('database/absensi.db')
cursor = conn.cursor()

# Buat tabel kalau belum ada
cursor.execute('''
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nama TEXT NOT NULL,
    encoding TEXT NOT NULL
)
''')
conn.commit()

# Inisialisasi kamera
cam = cv2.VideoCapture(0)

nama = input("Masukkan nama: ")
print("Arahkan wajah ke kamera, tekan 's' untuk menyimpan!")

while True:
    ret, frame = cam.read()
    if not ret:
        break

    cv2.imshow("Register Wajah", frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        # Simpan satu frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Cari lokasi wajah
        wajah = face_recognition.face_locations(rgb_frame)

        if len(wajah) == 0:
            print("Tidak ada wajah terdeteksi, coba lagi!")
            continue

        # Buat encoding
        encoding = face_recognition.face_encodings(rgb_frame, wajah)[0]

        # Simpan ke database
        cursor.execute("INSERT INTO faces (nama, encoding) VALUES (?, ?)", (nama, json.dumps(encoding.tolist())))
        conn.commit()
        print(f"Data wajah {nama} berhasil disimpan!")

        break

cam.release()
cv2.destroyAllWindows()
conn.close()
print("Proses pendaftaran wajah selesai.")