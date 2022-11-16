import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from config.config import DATA_DIR, STORE_DIR
from otracking.yolo import YOLO
from otracking.utils import draw, read_yolo_labels

model_dir = Path(STORE_DIR, "models", "yolov3")

labels = read_yolo_labels(model_dir)
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

yolo_model = YOLO("yolov3")

img_path = DATA_DIR / "mall_2.jpg"#"crow_night.png"# "seq_000024.jpg" # "cars.png"
img = cv2.imread(str(img_path))

boxes, classes, scores = yolo_model.predict(img)

img = draw(img, boxes, classes, scores, labels, colors)

cv2.imshow("mall", img)
cv2.imwrite(str(DATA_DIR/"mall_2_out.jpg"), img)
