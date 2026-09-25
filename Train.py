from config import *
from recognizer import *

def load_dataset(dataset_dir=DATASET_DIR):
    if not os.path.isdir(dataset_dir):
        raise IOError(f'Directory {dataset_dir} not found')

    class_names = list()
    for i in os.listdir(dataset_dir):
        if os.path.isdir(os.path.join(dataset_dir, i)):
            class_names.append(i)

    class_names = sorted(class_names)
    if len(class_names) < MIN_CLASSES:
        raise ValueError(f'Directory {dataset_dir} has {len(class_names)} subdirectories')

    images = list()
    labels = list()

    for idx, name in enumerate(class_names):
        folder = os.path.join(dataset_dir, name)
        files = list()

        for i in sorted(os.listdir(folder)):
            if i.lower().endswith(IMG_EXT):
                files.append(i)

        if not files:
            raise ValueError(f'Directory {folder} is empty')

        n = 0

        for f in files:
            path = os.path.join(folder, f)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f" !!! Unreadable file was skipped, {f} !!! ")
                continue

            images.append(to_vector(img))
            labels.append(idx)
            n += 1

        if n < MIN_PER_CLASS:
            loaded = n
            raise ValueError(
                f'В классе {name} загружено {loaded} фото, '
                f'минимум {MIN_PER_CLASS} (каталог: {folder})'
            )

        print(f'  {name:>8} (id={idx}): {n} foto')

    return np.array(images, np.float32), np.array(labels, dtype=np.int32), class_names

def train_test_split(x, y, test_size=TEST_SIZE, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    train_idx, test_idx = list(), list()

    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_size))
        n_test = max(1, min(n_test, len(idx)-1))

        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])

    test_idx = np.array(sorted(test_idx), dtype=np.int32)
    train_idx = np.array(sorted(train_idx), dtype=np.int32)

    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]

def fit_centroids(x_train, y_train, n_classes) -> np.ndarray:
    counts = np.bincount(y_train, minlength=n_classes)

    if (counts == 0).any():
        missing_classes = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"Пропавшие классы: \n{missing_classes}")

    x_train_norm = l2_normalize(x_train)
    centroids = np.stack([x_train_norm[y_train == i].mean(axis=0) for i in range(n_classes)])
    return l2_normalize(centroids).astype(np.float32)

def similarities(x, centroids) -> np.ndarray:
    return l2_normalize(x)@centroids.T

def predict_batch(x, centroids):
    sims = similarities(x, centroids)
    return np.argmax(sims, axis=1).astype(np.int32), np.max(sims, axis=1).astype(np.float32)

def fit_threshold(x_train, y_train, centroids) -> float:
    own = (l2_normalize(x_train) * centroids[y_train]).sum(axis=1)
    return float(np.percentile(own, THRESHOLD_PERCENTILE))

def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int32)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm

def print_report(y_true, y_pred, scores, threshold, class_names, centroids) -> float:
    cm = confusion_matrix(y_true, y_pred, len(class_names))
    accuracy = float((y_pred == y_true).mean())
    per_class = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    w = max(8, max(map(len, class_names)) + 2)

    print(f'\nthreshold: {threshold:.4f}')
    print(f'accuracy:  {accuracy:.4f}')
    print(f'rejected by threshold: {float((scores < threshold).mean()):.3f}')

    print('\nconfusion matrix (строки — истина, столбцы — предсказание)')
    print(''.join(f'{h:>{w}}' for h in [''] + list(class_names)))
    for name, row, acc in zip(class_names, cm, per_class):
        cells = ''.join(f'{v:>{w}}' for v in row)
        print(f'{name:>{w}}{cells}   acc {acc:.3f}')

    if len(class_names) == 2:
        print(f'\nsimilarity between centroids: {float(centroids[0] @ centroids[1]):.4f}')

    return accuracy

def save_model(path, centroids, class_names, threshold, accuracy):
    np.savez(
        path,
        centroids=centroids,
        class_names=np.array(class_names),
        face_size=np.array(FACE_SIZE, dtype=np.int32),
        threshold=np.float32(threshold),
        metric=np.array(METRIC),
        accuracy=np.float32(accuracy),
    )
    print(f'\nмодель сохранена: {path}')

def main():
    print('Загрузка датасета:')
    x, y, class_names = load_dataset()
    print(f'всего: {len(x)} фото, вектор: {x.shape[1]} признаков')

    x_train, x_test, y_train, y_test = train_test_split(x, y)
    print(f'train: {len(x_train)}   test: {len(x_test)}')

    centroids = fit_centroids(x_train, y_train, len(class_names))
    threshold = fit_threshold(x_train, y_train, centroids)

    y_pred, scores = predict_batch(x_test, centroids)
    accuracy = print_report(y_test, y_pred, scores, threshold, class_names, centroids)

    save_model(MODEL_PATH, centroids, class_names, threshold, accuracy)


if __name__ == '__main__':
    main()