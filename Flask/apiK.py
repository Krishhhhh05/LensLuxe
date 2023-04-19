import flask
import io
import string
import time
import os
import math
import time
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, jsonify, request
from IPython import display
display.clear_output()
import ultralytics
ultralytics.checks()
import cv2
from ultralytics import YOLO
from IPython.display import display, Image
import torch
import urllib
import requests
model = torch.load('/Users/krishangshah/PycharmProjects/pythonProject/venv/gender.pt')

f=None
def prepare_image(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    # img = cv2.imread("/content/profile9.jpg")
    img = img
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.01, minNeighbors=4)

    if len(faces)==0:
        print("hi")
        #print("NO FACE DETECTED")
        f=0



def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
            cropped = frameOpencvDnn[y1:y2+y2, x1:x2+x2]
            prepare_image(cropped)
    return frameOpencvDnn, bboxes

faceProto = "modelNweight/opencv_face_detector.pbtxt"
faceModel = "modelNweight/opencv_face_detector_uint8.pb"

ageProto = "modelNweight/age_deploy.prototxt"
ageModel = "modelNweight/age_net.caffemodel"

genderProto = "modelNweight/gender_deploy.prototxt"
genderModel = "modelNweight/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load network
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
faceNet = cv2.dnn.readNet(faceModel, faceProto)

padding = 20

def age_gender_detector(frame):
    # Read frame
    t = time.time()
    frameFace, bboxes = getFaceBox(faceNet, frame)
    #print(bboxes)
    if len(bboxes)==0:
        #print("fuck")
        return None,None
    for bbox in bboxes:
        # print(bbox)
        face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
               max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        label = "{},{}".format(gender, age)

        cv2.putText(frameFace, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                   cv2.LINE_AA)
    return frameFace,label


# input = cv2.imread("/Users/krishangshah/Downloads/trinity_hack_dataset/profile8.jpeg")
# output=age_gender_detector(input)
# print(output)
# print(f)
# if output is None or f is None:
#     print("NO FACES")
# else:
#     cv2.imshow("output",output)
#     cv2.waitKey(0)

def main(url):
   req = urllib.request.urlopen(url)
   arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
   input = cv2.imdecode(arr, -1)
   #cv2.imshow("dgdghd", img)
   output,label = age_gender_detector(input)
   #print(output)
   #print(f)
   if output is None and f is None:
       print("NO FACES")
   else:
       print(label)



app = Flask(__name__)


@app.route('/predict', methods=['POST','GET'])
def infer_image():

    print(request.args['url'])
    main(request.args['url'])
    if 'file' not in request.files:
        return ""

    file = request.files.get('file')

    if not file:
        return

    # Read the image
    img_bytes = file.read()

    # Prepare the image
    img = prepare_image(img_bytes)

    # Return on a JSON format
    return jsonify(prediction=predict_result(img))


@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

