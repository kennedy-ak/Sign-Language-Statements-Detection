import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3
import time  # Import the time module

# Initialize the TTS engine
engine = pyttsx3.init()

# Set properties for the TTS engine
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300
labels = ["Sorry", "Thank You", "Yes", "I love you", "Hello"]

# Minimum delay between detections (in seconds)
detection_delay = 2.0
last_detection_time = 0

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        continue

    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        current_time = time.time()
        # Check if enough time has passed since the last detection
        if current_time - last_detection_time > detection_delay:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            # Ensure the cropping coordinates are within image bounds
            y1 = max(0, y - offset)
            y2 = min(img.shape[0], y + h + offset)
            x1 = max(0, x - offset)
            x2 = min(img.shape[1], x + w + offset)

            imgCrop = img[y1:y2, x1:x2]
            imgCropShape = imgCrop.shape

            if imgCropShape[0] > 0 and imgCropShape[1] > 0:  # Ensure imgCrop is not empty
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap: wCal + wGap] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)

                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap: hCal + hGap, :] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)

                cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50),
                              (0, 255, 0),
                              cv2.FILLED)

                cv2.putText(imgOutput, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

                cv2.imshow('ImageCrop', imgCrop)
                cv2.imshow('ImageWhite', imgWhite)

                # Speak the predicted label
                engine.say(labels[index])
                engine.runAndWait()

                # Update the last detection time
                last_detection_time = current_time

    cv2.imshow('Image', imgOutput)
    cv2.waitKey(1)
