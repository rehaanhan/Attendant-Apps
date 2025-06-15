import sqlite3
import cv2
from datetime import datetime
from tensorflow.keras.models import load_model
import numpy as np

# Load model CNN wajah dan mood
model_wajah = load_model('model/face_recognition_cnn.h5')
model_mood = load_model('model/emotion_model.keras')  # model emosi kamu

print(f"Jumlah label di model wajah: {model_wajah.output_shape[-1]}")


# Label hasil training
labels_wajah = ['rehan', 'engkay']  # sesuaikan
labels_mood = ['Marah', 'Jijik', 'Takut', 'Senang', 'Sedih', 'Terkejut', 'Netral']  # sesuai 7 kelas kamu

print(f"Jumlah label di labels_wajah: {len(labels_wajah)}")


# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Koneksi ke SQLite
# conn = sqlite3.connect('database/absensi.db')
# cursor = conn.cursor()

# Kamera
cam = cv2.VideoCapture(0)
print("Mulai absensi dan deteksi mood... Tekan 'q' untuk keluar.")

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
       # Untuk prediksi wajah (model_wajah)
        face_color = frame[y:y+h, x:x+w]
        face_color_resized = cv2.resize(face_color, (160, 160))  # sesuai training
        face_color_array = face_color_resized.astype('float32') / 255.0
        face_color_array = np.expand_dims(face_color_array, axis=0)


        # Proses untuk model mood (grayscale, 48x48)
        face_gray = gray[y:y+h, x:x+w]
        try:
            face_gray_resized = cv2.resize(face_gray, (48, 48))
        except:
            continue
        face_gray_array = face_gray_resized.astype('float32') / 255.0
        face_gray_array = np.expand_dims(face_gray_array, axis=0)
        face_gray_array = np.expand_dims(face_gray_array, axis=-1)

        # Prediksi wajah
        pred_wajah = model_wajah.predict(face_color_array, verbose=0)[0]
        label_idx_wajah = np.argmax(pred_wajah)
        confidence_wajah = pred_wajah[label_idx_wajah]

        # Prediksi mood
        pred_mood = model_mood.predict(face_gray_array, verbose=0)[0]
        label_idx_mood = np.argmax(pred_mood)
        mood = labels_mood[label_idx_mood]

        # Hanya absensi jika confidence wajah tinggi
        if confidence_wajah > 0.8:
            name = labels_wajah[label_idx_wajah]
            waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # cursor.execute("INSERT INTO absensi (nama, waktu, mood) VALUES (?, ?, ?)", (name, waktu, mood))
            # conn.commit()
            print(f"Absensi {name} ({mood}) tercatat pada {waktu}")
        else:
            name = "Tidak Dikenal"

        # Tampilkan hasil di frame
        label_display = f"{name} | {mood}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label_display, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Absensi + Deteksi Mood", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
conn.close()
