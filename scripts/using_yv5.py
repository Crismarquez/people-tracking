import cv2
import matplotlib.pyplot as plt
import numpy as np

from config.config import DATA_DIR
from otracking.yolo import Yolov5

model = Yolov5("yv5_onnx", 0.5)

img_name = "crow_2.png"
img = cv2.imread(str(DATA_DIR / img_name))

_, results = model.predict(img)

image = results.render()
image = np.squeeze(image, axis=0)

cv2.imwrite(str(DATA_DIR/("output_"+img_name)), img)
cv2.imshow("image", image)
cv2.waitKey(1)
cv2.destroyAllWindows()
