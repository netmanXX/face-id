from config import *
from recognizer import load_model, predict


def run_realtime_recognition():
    color = GREEN

    try:
        model = load_model()
        print(f"Модель загруженна, датасет загружен - {model['class_names']}")
    except (FileNotFoundError, ValueError) as e:
        raise IOError(f"Згрузите модельку, пж... Всего-то train запустить\n{e}")

    cap = open_camera()
    cascade = load_cascade()
    running = True

    while running:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detect_faces(cascade, gray)
        display = frame.copy()

        status_text = f"Faces in frame {len(faces)}"
        cv2.putText(display, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GRAY, 2)

        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x + w, y + h), GREEN if color == GREEN else RED, 2)

            cropped_face = crop_face(gray, (x, y, w, h))

            if cropped_face.size == (0, 0):
                continue

            class_name, score = predict(model, cropped_face)

            if class_name == 'unknow':
                text_display = f"unknown ({score:.2f})"
                color = RED
            else:
                text_display = f"{class_name} ({score:.2f})"
                color = GREEN

            cv2.putText(display, text_display, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow('Face ID', display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            running = False

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_realtime_recognition()
