###########################################################################
#      Computer Vision Project - Facial Expression classification         #
#    *Phase 3 - Face Detection and Emotion Classification in realtime*    #
#               Taha Samavati-9423993,Abbas Mostafanasab - 94             #
###########################################################################

from keras.preprocessing.image import img_to_array
import cv2
from keras.models import load_model
import numpy as np


# 500:Y=12 X=13
# Pie1Y=20 X=15

def predict_emotion(faces):
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
                       key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
        # Extract the ROI of the face from the grayscale image, resize it to a fixed 48x48 pixels, and then prepare
        # the ROI for classification via the CNN
        roi = gray[fY :fY + fH, fX :fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = (roi - 0.5) * 2
        roi = img_to_array(roi)
        # add 1 more dim to have (1,48,48,1) shape
        roi = np.expand_dims(roi, axis=0)

        preds = emotion_classifier.predict(roi)[0]
        label = EMOTIONS[preds.argmax()]
        # emotion_probability = np.max(preds)
        # emotion_probability_text="{0:.1f}".format(emotion_probability)

        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            # construct the label text
            text = "{}: {:.2f}%".format(emotion, prob * 100)
            w = int(prob * 300)
            # Display probs in horizontal bars
            cv2.rectangle(probs_window, (7, (i * 35) + 5),
                          (w, (i * 35) + 35), (255, 0, 0), -1)
            cv2.putText(probs_window, text, (10, (i * 35) + 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 2)
            cv2.putText(frameClone, label, (fX, fY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            # cv2.putText(frameClone, emotion_probability_text, (fX+50, fY - 10),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                          (0, 0, 255), 2)

    else:
        cv2.putText(frameClone, "Unable to detect face", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


# define model paths
# Note : Obviously Our trained cascade classifier doesn't perform as Open CV's pre-trained classifiers
# due to their huge amount of data and computational power But it's reliable.
# You can use haarcascade_frontalface_default.xml (opencv model)in order to examine our cnn model performance separately.
face_detection_model_path = './models/haarcascade/haarcascade_frontalface_default.xml'
emotion_model_path = './models/mini_XCEPTION.73-0.65.hdf5'

# load models
face_detection = cv2.CascadeClassifier(face_detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
# Labels
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised",
            "neutral"]

# start video capturing
cam = cv2.VideoCapture(0)
while True:
    frame = cam.read()[1]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                            flags=cv2.CASCADE_SCALE_IMAGE)

    probs_window = np.zeros((300, 300, 3), dtype="uint8")
    frameClone = frame.copy()

    predict_emotion(faces)

    cv2.imshow('emotion classification', frameClone)
    cv2.imshow("Probabilities", probs_window)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
