import numpy as np
import os
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(BASE_DIR, 'faceid.npz')
CASCADE_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')

if not os.path.exists(CASCADE_PATH):
    CASCADE_PATH = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')

# ---------- параметры ----------
FACE_SIZE = (160, 160)
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp')

SCALE_FACTOR = 1.2
MIN_NEIGHBORS = 6
MIN_FACE = (120, 120)
MARGIN = 0.2
PIXEL_MAX = 255.0
TEST_SIZE = 0.2
RANDOM_SEED = 42
EPS = 1e-8
METRIC = 'cosine'
THRESHOLD_PERCENTILE = 90
MIN_PER_CLASS = 10
MIN_CLASSES = 2
CAMERA_INDEX = 0
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0, 255, 0)
RED = (0, 0, 255)
GRAY = (200, 200, 200)
TEXT_X = 10

def load_cascade(path:str = CASCADE_PATH) -> cv2.CascadeClassifier:
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise IOError(' Не удалось загрузить каскад (1)')

    return cascade

def open_camera(index:int = CAMERA_INDEX) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise IOError(' Камера не найденна (2)')

    return cap

def detect_faces(cascade, gray) -> list[tuple[int, int, int, int]]:
    faces = cascade.detectMultiScale(gray, SCALE_FACTOR, MIN_NEIGHBORS, minSize=MIN_FACE)
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]

def crop_face(img, box, margin = MARGIN):
    x, y, w, h = box
    ih, iw = img.shape[:2]
    dx = int(w*margin)
    dy = int(h*margin)
    x1 = max(0, x-dx)
    y1 = max(0, y-dy)
    x2 = min(iw, x+w+dx)
    y2 = min(ih, y+h+dy)

    return img[y1:y2, x1:x2]

def preprocess(gray):
    face = cv2.resize(gray, FACE_SIZE, interpolation=cv2.INTER_LINEAR)
    return cv2.equalizeHist(face)

def put_lines(img, lines, color=GREEN):
    for i, text in enumerate(lines):
        cv2.putText(img, text, (TEXT_X, 25+i*25), FONT, 0.7, color, 2)