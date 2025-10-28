# Install required packages before running:
# pip install tensorflow pillow matplotlib numpy

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from PIL import Image
import numpy as np
import os

# ---------------------------
# Load & Preprocess Dataset
# ---------------------------

print("Loading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0

print("Training data:", x_train.shape)
print("Testing data:", x_test.shape)

# ---------------------------
# Build CNN Model
# ---------------------------

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ---------------------------
# Train & Save Model
# ---------------------------

print("\nTraining model...")
history = model.fit(x_train, y_train, epochs=10,
                    validation_data=(x_test, y_test))

# Evaluate accuracy
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print("\n✅ Test accuracy:", test_acc)

# Save trained model to file
model.save("cifar_model.h5")
print("✅ Model saved: cifar_model.h5")

# ---------------------------
# Prediction from Local Image
# ---------------------------

# CIFAR-10 class labels
class_names = [
    'airplane','automobile','bird','cat','deer',
    'dog','frog','horse','ship','truck'
]

while True:
    print("\nEnter an image file path (type 'exit' to quit):")
    image_path = input(">> ")

    if image_path.lower() == "exit":
        break

    if not os.path.exists(image_path):
        print("❌ File not found. Try again.")
        continue

    # Load and preprocess image
    img = Image.open(image_path).resize((32, 32))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_label = class_names[np.argmax(prediction)]

    print(f"✅ Prediction: {predicted_label}")