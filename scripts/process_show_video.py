from pathlib import Path
import cv2
import numpy as np

from config.config import DATA_DIR, STORE_DIR
from otracking.yolo import YOLO, Yolov5
from otracking.utils import draw, read_yolo_labels

model_dir = Path(STORE_DIR, "models", "yolov3")
path_video = DATA_DIR / "1_pasillo_mall.mp4"
path_out = DATA_DIR / "mall_out_video.mp4"

labels = read_yolo_labels(model_dir)
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

skip_fps = 1
threshold = 0.3

vs = cv2.VideoCapture(str(path_video))

writer = None

# Definimos ancho y alto
W = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Inicializamos variables principales
#yolo = YOLO("yolov3")
yolo = Yolov5("yv5_pt", 0.5)

totalFrame = 0
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
writer = cv2.VideoWriter(str(path_out), fourcc, 20.0, (W, H), True)
while True:
    # Leemos el primer frame
    ret, frame = vs.read()

    # Si ya no hay más frame, significa que el video termino y por tanto se sale del bucle
    if frame is None:
        break

    status = "Waiting"
    rects = []

    # Nos saltamos los frames especificados.
    if totalFrame % skip_fps == 0:
        # Predecimos los objectos y clases de la imagen
        # boxes, classes, scores = yolo.predict(frame)
        _, results = yolo.predict(frame)


    # frame = draw(frame, boxes, classes, scores, labels, colors)
    frame = results.render()
    frame = np.squeeze(frame, axis=0)

    totalFrame += 1

    cv2.imshow("people detector", frame)

    writer.write(frame)

    if cv2.waitKey(10) == ord("q"):
        break

vs.release()
writer.release()
cv2.destroyAllWindows()




