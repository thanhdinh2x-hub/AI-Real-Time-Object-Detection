from pathlib import Path
from PIL import Image

folder = Path("$PWD/datasets/VOCdevkit/VOC2007/JPEGImages")

bad_images = []

for path in folder.glob("*.jpg"):
    try:
        with Image.open(path) as img:
            img.load()
    except Exception as e:
        bad_images.append((str(path), str(e)))

print("Số ảnh lỗi:", len(bad_images))

for path, error in bad_images:
    print(path, "->", error)