import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LambdaCallback  # <-- added
from model import get_model

DATA_DIR = 'datasets/'
IMG_SIZE = 224
def load_data():
    X, y = [], []
    for label, folder in enumerate(['real', 'spoof']):
        path = os.path.join(DATA_DIR, folder)
        for img_file in os.listdir(path):
            img_path = os.path.join(path, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                X.append(img)
                y.append(label)
    return np.array(X), np.array(y)

X, y = load_data()
X = X / 255.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = get_model((IMG_SIZE, IMG_SIZE, 3))
datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=15)

log_callback = LambdaCallback(
    on_epoch_end=lambda epoch, logs: print(
        f"Epoch {epoch+1} — Loss: {logs['loss']:.4f}, "
        f"Acc: {logs['accuracy']:.4f}, "
        f"Val Loss: {logs['val_loss']:.4f}, "
        f"Val Acc: {logs['val_accuracy']:.4f}"
    )
)

model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_test, y_test),
    epochs=10,
    callbacks=[log_callback],
    verbose=0  
)

model.save("anti_spoof_model.h5")

