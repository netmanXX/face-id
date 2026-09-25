English
Overview

This project implements a lightweight face recognition pipeline:

    Collect face images via webcam (Collector.py)

    Train a model by computing class centroids (Train.py)

    Recognize faces in real-time (main.py)

The recognition method is straightforward: each face is converted to a normalized feature vector, and classification is performed by finding the nearest class centroid using cosine similarity.
Requirements

    Python 3.8+

    OpenCV (opencv-python)

    NumPy

Install dependencies:
bash

pip install opencv-python numpy

Project Structure
text

.
├── Collector.py          # Webcam tool for collecting face images
├── Train.py              # Model training script
├── main.py               # Real-time face recognition
├── recognizer.py         # Core recognition logic (vectorization, prediction)
├── config.py             # Shared constants and utility functions
├── Detector.py           # Simple face detector demo
├── rename.py             # Utility to normalize filenames in a dataset folder
├── haarcascade_frontalface_default.xml   # Haar cascade for face detection
├── dataset/              # Collected face images (created automatically)
│   ├── me/
│   └── not_me/
└── faceid.npz            # Trained model (created by Train.py)

Workflow
1. Collect Face Images

Run the collector and save face crops into dataset/<label>/:
bash

python Collector.py

Controls:
Key	Action
x	Save current face (only if exactly one face is detected)
z	Switch label (me ↔ not_me)
q / Esc	Quit

    Note: The current implementation automatically saves a face every frame when exactly one face is detected. If you want manual-only capture, remove the block that saves unconditionally.

Recommended: at least 10 images per class.
2. Train the Model
bash

python Train.py

The script will:

    Load all images from dataset/

    Convert each face to a normalized 160×160 grayscale vector

    Compute per-class centroids

    Determine a rejection threshold (90th percentile of intra-class similarities)

    Split data into train/test, evaluate, and print a confusion matrix

    Save the model to faceid.npz

3. Run Real-Time Recognition
bash

python main.py

A window will open with your webcam feed. Detected faces are labeled with either a known class name + similarity score, or unknown if the score falls below the threshold.

Press q or Esc to quit.
Configuration

All tunable parameters live in config.py:
Parameter	Description	Default
FACE_SIZE	Size face images are resized to	(160, 160)
SCALE_FACTOR	Haar cascade scale factor	1.2
MIN_NEIGHBORS	Haar cascade min neighbors	6
MIN_FACE	Minimum face size in pixels	(120, 120)
MARGIN	Crop margin around detected face	0.2
TEST_SIZE	Fraction of data for testing	0.2
THRESHOLD_PERCENTILE	Percentile for rejection threshold	90
MIN_PER_CLASS	Minimum images required per class	10
MIN_CLASSES	Minimum number of classes	2
CAMERA_INDEX	Webcam device index	0
How It Works

Feature extraction (recognizer.py):

    Resize face to FACE_SIZE

    Apply histogram equalization

    Flatten to a 1-D vector and scale by 1/255

    L2-normalize

Training (Train.py):

    For each class, compute the mean of its normalized vectors

    L2-normalize the resulting centroid

    Threshold = 90th percentile of cosine similarity between training samples and their own centroid

Prediction (recognizer.py):

    Compute cosine similarity between the input vector and every centroid

    Take the class with the highest similarity

    If the best similarity is below the threshold → return unknown

Notes & Limitations

    This is a baseline method using raw pixel features. It works well for small, controlled datasets but is not robust to lighting, pose, or expression changes.

    For better accuracy, consider extracting embeddings from a pretrained CNN (e.g., FaceNet, ArcFace).

    The Haar cascade is a classical detector — good enough for frontal faces, but slow and less accurate than modern detectors (e.g., DNN-based).

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Русский
Обзор

Проект реализует простой пайплайн распознавания лиц:

    Сбор изображений лиц через веб-камеру (Collector.py)

    Обучение модели — вычисление центроидов классов (Train.py)

    Распознавание лиц в реальном времени (main.py)

Метод распознавания прост: каждое лицо преобразуется в нормализованный вектор признаков, классификация выполняется поиском ближайшего центроида класса по косинусной близости.
Требования

    Python 3.8+

    OpenCV (opencv-python)

    NumPy

Установка зависимостей:
bash

pip install opencv-python numpy

Структура проекта
text

.
├── Collector.py          # Инструмент сбора изображений лиц с веб-камеры
├── Train.py              # Скрипт обучения модели
├── main.py               # Распознавание лиц в реальном времени
├── recognizer.py         # Ядро распознавания (векторизация, предсказание)
├── config.py             # Общие константы и утилиты
├── Detector.py           # Простая демонстрация детектора лиц
├── rename.py             # Утилита нормализации имён файлов в папке датасета
├── haarcascade_frontalface_default.xml   # Haar-каскад для детекции лиц
├── dataset/              # Собранные изображения (создаётся автоматически)
│   ├── me/
│   └── not_me/
└── faceid.npz            # Обученная модель (создаётся Train.py)

Порядок работы
1. Сбор изображений лиц

Запустите сборщик и сохраняйте вырезки лиц в dataset/<метка>/:
bash

python Collector.py

Управление:
Клавиша	Действие
x	Сохранить текущее лицо (только если в кадре ровно одно лицо)
z	Переключить метку (me ↔ not_me)
q / Esc	Выход

    Замечание: В текущей реализации лицо сохраняется автоматически каждый кадр, если в кадре ровно одно лицо. Если нужно сохранять только вручную, уберите блок безусловного сохранения.

Рекомендуется собрать не менее 10 изображений на класс.
2. Обучение модели
bash

python Train.py

Скрипт:

    Загрузит все изображения из dataset/

    Преобразует каждое лицо в нормализованный вектор 160×160 (grayscale)

    Вычислит центроиды по классам

    Определит порог отсечения (90-й перцентиль внутриклассовой близости)

    Разделит данные на train/test, оценит качество, напечатает матрицу ошибок

    Сохранит модель в faceid.npz

3. Распознавание в реальном времени
bash

python main.py

Откроется окно с видеопотоком с камеры. Обнаруженные лица помечаются именем класса и оценкой близости, либо unknown, если оценка ниже порога.

Выход — q или Esc.
Настройки

Все параметры находятся в config.py:
Параметр	Описание	По умолчанию
FACE_SIZE	Размер, к которому приводится лицо	(160, 160)
SCALE_FACTOR	Масштабный коэффициент Haar-каскада	1.2
MIN_NEIGHBORS	Минимум соседей Haar-каскада	6
MIN_FACE	Минимальный размер лица в пикселях	(120, 120)
MARGIN	Отступ при вырезании лица	0.2
TEST_SIZE	Доля данных для теста	0.2
THRESHOLD_PERCENTILE	Перцентиль для порога отсечения	90
MIN_PER_CLASS	Минимум изображений на класс	10
MIN_CLASSES	Минимум классов	2
CAMERA_INDEX	Индекс камеры	0
Как это работает

Извлечение признаков (recognizer.py):

    Масштабирование лица до FACE_SIZE

    Эквализация гистограммы

    Разворачивание в одномерный вектор и нормировка на 1/255

    L2-нормализация

Обучение (Train.py):

    Для каждого класса — среднее нормализованных векторов

    L2-нормализация полученного центроида

    Порог = 90-й перцентиль косинусной близости между обучающими примерами и их центроидом

Предсказание (recognizer.py):

    Косинусная близость между входным вектором и каждым центроидом

    Берётся класс с максимальной близостью

    Если лучшая близость ниже порога → возвращается unknown

Замечания и ограничения

    Это базовый метод на сырых пиксельных признаках. Хорошо работает на небольших контролируемых датасетах, но неустойчив к освещению, позе и мимике.

    Для повышения точности стоит использовать эмбеддинги из предобученных CNN (например, FaceNet, ArcFace).

    Haar-каскад — классический детектор: подходит для фронтальных лиц, но медленнее и менее точен, чем современные (например, DNN-based).
