from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import io
import string
import time
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import requests
import cv2
from ultralytics import YOLO
import torch
import uvicorn

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = os.path.abspath(os.path.join(__file__ ,"Flask\gender.pt"))
print(model_path)
model = torch.load(model_path)

def prepare_image(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.01, minNeighbors=4)
    if len(faces) == 0:
        return None
    for x, y, w, h in faces:
        cropped = img[y:y + h , x:x + w ]
    return gen(cropped)

def gen(img):
    l = []
    model = YOLO(model_path)
    results = model.predict(img, conf=0.06)
    for i in range(0, len(results[0].boxes.xyxy)):
        k = results[0].boxes.xyxy[i].numpy()
        if results[0].boxes.cls[i].numpy() == 1:
            l.append("Male")
        else:
            l.append("Female")
    if len(l) == 0:
        l.append("Unknown")
    return max(set(l), key = l.count)

@app.post("/predict")
async def predict_gender(url: str):
    img_response = requests.get(url)
    img_bytes = io.BytesIO(img_response.content)
    img = cv2.imdecode(np.frombuffer(img_bytes.getvalue(), np.uint8), cv2.IMREAD_UNCHANGED)
    prediction = prepare_image(img)
    if prediction is None:
        prediction = "Human Face not detected"
    return {"prediction": prediction}

@app.get("/")
async def index():
    return {"message": "Machine Learning Inference"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
