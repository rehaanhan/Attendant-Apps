import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import mysql.connector
from datetime import datetime


# Path ke dataset dan model
DATASET_DIR = 'dataset/wajah'
MODEL_DIR = 'model'
MODEL_NAME = 'face_recognition_cnn.h5'

# Ukuran gambar yang konsisten
IMG_SIZE = (160, 160)
BATCH_SIZE = 8
EPOCHS = 10

# Data augmentation dan normalisasi
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Model CNN sederhana
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# Simpan model
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

model.save(os.path.join(MODEL_DIR, MODEL_NAME))
print(f"Model saved to {os.path.join(MODEL_DIR, MODEL_NAME)}")
