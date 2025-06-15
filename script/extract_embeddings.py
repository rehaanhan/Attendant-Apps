import os
import numpy as np
import pickle
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from utils import extract_face
import cv2

model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))

dataset_dir = 'dataset/wajah'
embeddings = []
labels = []

for person_name in os.listdir(dataset_dir):
    person_path = os.path.join(dataset_dir, person_name)
    if not os.path.isdir(person_path):
        continue

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        face = extract_face(img)
        if face is not None:
            face = np.expand_dims(face, axis=0)
            feature = model.predict(face)[0]
            embeddings.append(feature)
            labels.append(person_name)
        else:
            print(f"Wajah tidak terdeteksi: {img_path}")

# Simpan embeddings ke file
with open("embeddings.pkl", "wb") as f:
    pickle.dump((embeddings, labels), f)

print("[INFO] Embeddings berhasil disimpan ke embeddings.pkl")
