import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model

# Load model CNN emosi
model = load_model("model/emotion_model.keras")

# Label emosi
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load detektor wajah CNN (MTCNN)
detector = MTCNN()

# Buka kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_frame)

    for face in faces:
        x, y, width, height = face['box']
        x, y = abs(x), abs(y)

        # Crop wajah
        face_img = frame[y:y+height, x:x+width]
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = cv2.resize(face_img, (48, 48))
        face_img = face_img.astype('float32') / 255.0
        face_img = np.expand_dims(face_img, axis=-1)  # Tambah channel
        face_img = np.expand_dims(face_img, axis=0)   # Tambah batch size

        # Prediksi emosi
        prediction = model.predict(face_img)
        emotion = emotion_labels[np.argmax(prediction)]

        # Gambar kotak dan label
        cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)


    # Tampilkan hasil
    cv2.imshow('CNN Face and Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()
