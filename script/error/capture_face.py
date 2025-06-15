import cv2
import os

# Fungsi untuk membuat folder user jika belum ada
def make_dataset_dir(name):
    path = os.path.join('dataset/wajah', name)
    os.makedirs(path, exist_ok=True)
    return path

def capture_faces(name):
    save_path = make_dataset_dir(name)
    cam = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print("[INFO] Tekan 'q' untuk keluar.")
    print(f"[INFO] Menyimpan wajah untuk: {name}")

    count = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = frame[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (160, 160))
            filename = os.path.join(save_path, f'{count}.jpg')
            cv2.imwrite(filename, face_img)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Capture Faces', frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Selesai menyimpan {count} gambar ke {save_path}")

if __name__ == "__main__":
    name = input("Masukkan nama orang yang akan diambil wajahnya: ")
    capture_faces(name)
