from config import *

LABELS = ('me', 'not_me')

def save_face(clin_frame, box, outdir, index):
    face = crop_face(clin_frame, box)
    face = cv2.resize(face, FACE_SIZE, interpolation=cv2.INTER_AREA)

    file_path = os.path.join(outdir, f'face_{index:04d}.jpeg')

    cv2.imwrite(file_path, face, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return file_path

def out_dir_for(label):
    path = os.path.join(DATASET_DIR, label)
    os.makedirs(path, exist_ok=True)

    return path

def next_index(out_dir):
    biggest = -1
    for i in os.listdir(out_dir):
        stem, ext =  os.path.splitext(i)
        if ext.lower() in IMG_EXT and stem.startswith('face_'):
            digits = stem[len('face_'):]
            if digits.isdigit():
                biggest = max(biggest, int(digits))

    return biggest + 1

def collect():
    cascade = load_cascade()
    cap = open_camera()

    li = 0
    label = LABELS[li]
    out_dir = out_dir_for(label)
    count = next_index(out_dir)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(cascade, gray)

        display = frame.copy()
        for (fx, fy, fw, fh) in faces:
            cv2.rectangle(display, (fx, fy), (fx + fw, fy + fh), GREEN, 2)

        put_lines(display, [
            f'Faces in frame: {len(faces)}',
            f'label: {label}   saved: {count}',
            'x - save | z - switch | q - quit',
        ])
        cv2.imshow('Collector', display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

        if key == ord('x'):
            if len(faces) == 1:
                path = save_face(frame, faces[0], out_dir, count)
                print('сохранено:', path)
                count += 1
            else:
                print(f'В кадре должно быть ровно одно лицо, сейчас {len(faces)}')

        if len(faces) == 1:
            save_face(frame, faces[0], out_dir, count)
            count += 1

        if key == ord('z'):
            li = (li + 1) % len(LABELS)
            label = LABELS[li]
            out_dir = out_dir_for(label)
            count = next_index(out_dir)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    collect()