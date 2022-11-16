import os
from pathlib import Path
import cv2
import datetime
import numpy as np
from imutils.video import FPS

from otracking.yolo import Yolov5
from otracking.tracking import Tracker, TrackableObject 
from otracking.utils import draw, xyxy_to_xywh, to_center_objects
from config.config import DATA_DIR, STORE_DIR, CONFIG_DIR

import sys
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
if str(ROOT / 'otracking') not in sys.path:
    sys.path.append(str(ROOT / 'otracking')) 

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative



path_video = DATA_DIR / "v_crow.mp4"
path_out = DATA_DIR / "crow_out_video.mp4"


# initialize StrongSORT
config_strongsort = CONFIG_DIR / 'strong_sort.yaml'
strong_sort_weights = STORE_DIR / "models/strong_sort/osnet_x0_25_msmt17.pt"
device = "cpu"
half = False

tracker  = Tracker()

results = []

# Load model
yolo = Yolov5("yv5_pt", 0.5)

path_video = DATA_DIR / "1_pasillo_mall.mp4"
path_out = DATA_DIR / "out_video.mp4"

skip_fps = 30

vs = cv2.VideoCapture(str(path_video))

trackableObjects = {}
writer = None

# Definimos ancho y alto
W = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))

DIRECTION_PEOPLE = False

fps = FPS().start()

# Creamos un umbral para sabre si el carro paso de izquierda a derecha o viceversa
POINT = [0, int((H/2)-H*0.1), W, int(H*0.1)]


# Definimos el formato del archivo resultante y las rutas.
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
writer = cv2.VideoWriter(str(path_out), fourcc, 30.0, (W, H), True)


trackers = []
trackableObjects = {}
raw_data = {}
objects = {}

totalFrame = 0
totalDown = 0
totalUp = 0

while True:

    ret, frame = vs.read()

    if frame is None:
        break

    status = "Waiting"
    rects = []

    date_time = datetime.datetime.now()
    # Nos saltamos los frames especificados.
    crop_result, results = yolo.predict(frame)

    ext_results = results.pandas().xyxy[0]
    xyxy = torch.tensor(ext_results[['xmin', 'ymin', 'xmax', 'ymax']].values)
    confidences = torch.tensor(ext_results[['confidence']].values)
    clss = torch.tensor(ext_results[['class']].values)

    xywh= xyxy_to_xywh(xyxy)

    # get id and centroids dict
    objects = tracker.update(xywh, confidences, clss, frame)

    objects_to_save = {}
    # Recorremos cada una de las detecciones
    for (objectID, centroid) in objects.items():
        # Revisamos si el objeto ya se ha contado
        to = trackableObjects.get(objectID, None)
        if to is None:
            to = TrackableObject(objectID, centroid)

        else:
            # Si no se ha contado, analizamos la dirección del objeto
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)
            if not to.counted:
                if centroid[0] > POINT[0] and centroid[0] < (POINT[0]+ POINT[2]) and centroid[1] > POINT[1] and centroid[1] < (POINT[1]+POINT[3]):
                    if DIRECTION_PEOPLE:
                        if direction >0:
                            totalUp += 1
                            to.counted = True
                        else:
                            totalDown +=1
                            to.counted = True
                    else:
                        if direction <0:
                            totalUp += 1
                            to.counted = True
                        else:
                            totalDown +=1
                            to.counted = True

        trackableObjects[objectID] = to
        objects_to_save[objectID] = centroid

        # Dibujamos el centroide y el ID de la detección encontrada
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0]-5, centroid[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0,255,0), -1)

    if objects_to_save:
        raw_data[str(date_time)] = objects_to_save

    info = [
            ("Sur", totalUp),
            ("Norte", totalDown),
            ("Estado", status),
    ]

    for (i, (k,v)) in enumerate(info):
        text = "{}: {}".format(k,v)
        cv2.putText(frame, text, (10, H - ((i*20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    cv2.imshow("people detector", frame)

    writer.write(frame)

    if cv2.waitKey(10) == ord("q"):
        break

    # Almacenamos el framme en nuestro video resultante.
    writer.write(frame)
    totalFrame += 1
    fps.update()

# Terminamos de analizar FPS y mostramos resultados finales
fps.stop()

print("Tiempo completo {}".format(fps.elapsed()))
print("Tiempo aproximado por frame {}".format(fps.fps()))

# Cerramos el stream de consumir el video.
vs.release()


