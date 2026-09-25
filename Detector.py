import numpy as np
from config import *

CASCADE_PATH = "haarcascade_frontalface_default.xml"

cap = open_camera(0)
cascade = load_cascade(CASCADE_PATH)
running = True

def run():
    while running:
        ok, frame = cap.read()
        h, w = frame.shape[:2]

        if not ok:
            frame = np.zeros((h, w, 3), np.uint8)

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(cascade, gray)
        display = frame.copy()

        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(frame, f'Faces in frame: {len(faces)}', (10, 25), cv2.FONT_ITALIC,
                    0.7, (0, 255, 0), 2)

        cv2.imshow('Face ID', display)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            return