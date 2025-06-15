import cv2
import os
from utils import detect_face

name = input("Masukkan nama orang yang ingin didaftarkan: ").strip()

save_dir = os.path.join("dataset/wajah", name)
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0
max_images = 100

print("[INFO] Mulai pendaftaran wajah. Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detect_face(frame)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (224, 224))

        file_path = os.path.join(save_dir, f"{name}_{count}.jpg")
        cv2.imwrite(file_path, face_img)
        count += 1

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{count}/{max_images}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Daftar Wajah", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= max_images:
        break

cap.release()
cv2.destroyAllWindows()

print(f"[INFO] Pendaftaran selesai. {count} gambar disimpan di {save_dir}")
