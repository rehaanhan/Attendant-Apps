import cv2
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import MobileNetV2
from utils import extract_face, detect_face

# Load model & embeddings
model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
with open("embeddings.pkl", "rb") as f:
    known_embeddings, known_labels = pickle.load(f)

cap = cv2.VideoCapture(0)
print("[INFO] Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detect_face(frame)
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        processed_face = extract_face(face)

        if processed_face is not None:
            processed_face = np.expand_dims(processed_face, axis=0)
            embedding = model.predict(processed_face)[0]

            similarities = cosine_similarity([embedding], known_embeddings)[0]
            max_sim = np.max(similarities)
            max_idx = np.argmax(similarities)

            name = known_labels[max_idx] if max_sim > 0.7 else "Unknown"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({max_sim:.2f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
