import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("anti_spoof_model.h5")
IMG_SIZE = 224

st.title("Face Anti-Spoofing Real-Time Detection")
run = st.checkbox("📷 Start Camera")

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    face = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    face = np.expand_dims(face / 255.0, axis=0)
    pred = model.predict(face)[0][0]

    label = "Real" if pred < 0.5 else "Spoof"
    color = (0, 255, 0) if label == "Real" else (0, 0, 255)
    cv2.putText(frame, label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
else:
    cap.release()
