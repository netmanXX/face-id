import os

FOLDER = 'dataset/not_me'
list_dir = sorted(os.listdir(FOLDER))

for i, name in enumerate(list_dir):
    os.rename(os.path.join(FOLDER, name), os.path.join(FOLDER, f"tmp_{i:04d}"))

for i, name in enumerate(list_dir):
    os.rename(os.path.join(FOLDER, f"tmp_{i:04d}"), os.path.join(FOLDER, f"face_{i:04d}.jpg"))