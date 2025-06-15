import cv2
import numpy as np
import pickle
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics.pairwise import cosine_similarity

# ========== SETUP ==========
# Load model emosi
emotion_model = load_model("model/emotion_model.keras")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load model face embedding
face_embedder = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))

# Load data face embeddings
with open("embeddings.pkl", "rb") as f:
    known_embeddings, known_names = pickle.load(f)

# Load face detector (MTCNN)
detector = MTCNN()

# ========== FUNGSI ==========
def get_face_embedding(face_img):
    face = cv2.resize(face_img, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)
    embedding = face_embedder.predict(face)[0]
    return embedding

def recognize_face(embedding):
    similarities = cosine_similarity([embedding], known_embeddings)[0]
    max_sim = np.max(similarities)
    if max_sim > 0.7:
        return known_names[np.argmax(similarities)], max_sim
    return "Unknown", max_sim

# ========== KAMERA ==========
cap = cv2.VideoCapture(0)
print("[INFO] Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_frame)

    for face in faces:
        x, y, w, h = face['box']
        x, y = abs(x), abs(y)

        face_crop = frame[y:y+h, x:x+w]

        # ==== Face Recognition ====
        try:
            embed = get_face_embedding(face_crop)
            name, sim = recognize_face(embed)
        except:
            name, sim = "Unknown", 0

        # ==== Emotion Detection ====
        try:
            face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(face_gray, (48, 48))
            face_normalized = face_resized.astype('float32') / 255.0
            face_input = np.expand_dims(face_normalized, axis=(0, -1))

            prediction = emotion_model.predict(face_input)
            emotion = emotion_labels[np.argmax(prediction)]
        except:
            emotion = "?"
            

        # ==== Tampilan ====
        label = f"{name} - {emotion}"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255,255,255), 2)
        
        

    cv2.imshow("Face Recognition + Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
