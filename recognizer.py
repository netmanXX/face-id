from config import *

def face_to_vector(gray_face: np.ndarray) -> np.ndarray:
    return l2_normalize(to_vector(gray_face))

def to_vector(gray: np.ndarray) -> np.ndarray:
    face = preprocess(gray)
    ravel = face.ravel()

    return ravel.astype(np.float32) / PIXEL_MAX

def l2_normalize(x):
    if x.ndim == 1:
        return x / (np.linalg.norm(x) + EPS)

    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + EPS)

def load_model(path: str = MODEL_PATH) -> dict:
    try:
        with np.load(path, allow_pickle=False) as data:
            expected_keys = ['centroids', 'class_names', 'face_size', 'threshold', 'metric', 'accuracy']
            for key in expected_keys:
                if key not in data.files:
                    raise IOError(f"Поход нет ключика - {key}'")

            centroids = data['centroids']
            class_names = data['class_names']
            face_size = data['face_size']
            threshold = data['threshold']
            metric = data['metric']
            accuracy = data['accuracy']

            face_size = tuple(int(i) for i in data["face_size"])
            if face_size != FACE_SIZE:
                raise ValueError(
                    f"Неподходящий размер лица ({face_size}), требуется - {FACE_SIZE}"
                )

            centroids = centroids.astype(np.float32)
            class_names = list(map(str, class_names))

            threshold = float(data['threshold'])
            accuracy = float(data['accuracy'])
            metric = str(data['metric'])

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Модель не найденна, {path}. Для штатной работы надо б её сначала создать, нет?"
        )

    n_classes = len(class_names)
    expected_centroid_dim = (n_classes, FACE_SIZE[0] * FACE_SIZE[1])

    if centroids.shape != expected_centroid_dim:
        raise ValueError(f"Несоответствует форма {centroids.shape}")

    return {
        'centroids': centroids,
        'class_names': class_names,
        'face_size': np.array(FACE_SIZE, dtype=np.int32),
        'threshold': threshold,
        'metric': metric,
        'accuracy': accuracy,
    }

def predict(model: dict, gray_face: np.ndarray) -> tuple[str, float]:
    vec = face_to_vector(gray_face)
    sims = model['centroids']@vec
    best = int(np.argmax(sims))
    score = float(sims[best])
    if score < model['threshold']:
        return ('unknow', score)
    return (model['class_names'][best], score)