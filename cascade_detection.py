###########################################################################
#      Computer Vision Project - Facial Expression classification         #
#                   *Phase 1 - Cascade Face Detection *                   #
#                             Taha Samavati                               #
###########################################################################

import cv2
import numpy as np

# Note:we have trained couple of models , other models are also kept in same directory.
face_detection_model_path = './models/haarcascade/trained_cascade_500.xml'
face_detector = cv2.CascadeClassifier(face_detection_model_path)

# change to TRUE if you want to evaluate model on webcam video.
FLAG_VIDEO=True

if FLAG_VIDEO:

    cap = cv2.VideoCapture(0)

    while 1:
        ret, I = cap.read()
        gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(gray, 1.1,5)
        for (x, y, w, h) in faces:
            # Draw rectangle over face
            cv2.rectangle(I, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(I, "Face", (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 2)

        cv2.imshow('Cascade Detection', I)
        key = cv2.waitKey(50)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

else:
    cap = cv2.VideoCapture(0)

    for i in range(12):
        I = cap.read()[1]

    gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in faces:
        # Draw rectangle over face
        cv2.rectangle(I, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(I, "Face found", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 2)

    cv2.imshow('Cascade Detection', I)
    key = cv2.waitKey(0)
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
