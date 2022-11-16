from pathlib import Path
import json
import cv2
import numpy as np
from imutils.video import FPS
import dlib
import datetime

from config.config import DATA_DIR, STORE_DIR
from otracking.yolo import YOLO
from otracking.trackingeng import centroidtracker, trackableobject

model_dir = Path(STORE_DIR, "models", "yolov3")
path_video = DATA_DIR / "v_crow.mp4"
path_out = DATA_DIR / "crow_out_video.mp4"

camera_location = "pasillo_2"
period_time = "2022-08-05-10am_2022-08-05-11am"

skip_fps = 30
threshold = 0.3

vs = cv2.VideoCapture(str(path_video))

writer = None

# Definimos ancho y alto
W = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))

ct = centroidtracker.CentroidTracker(maxDisappeared= 40, maxDistance = 50)

# Inicializamos variables principales
yolo = YOLO("yolov3")
trackers = []
trackableObjects = {}
raw_data = {}

totalFrame = 0
totalDown = 0
totalUp = 0

DIRECTION_PEOPLE = True

# Creamos un umbral para sabre si el carro paso de izquierda a derecha o viceversa
# En este caso lo deje fijo pero se pudiese configurar según la ubicación de la cámara.
POINT = [0, int((H/2)-H*0.1), W, int(H*0.1)]

# Los FPS nos van a permitir ver el rendimiento de nuestro modelo y si funciona en tiempo real.
fps = FPS().start()

# Definimos el formato del archivo resultante y las rutas.
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
writer = cv2.VideoWriter(str(path_out), fourcc, 20.0, (W, H), True)

# Bucle que recorre todo el video
while True:
  # Leemos el primer frame
  ret, frame = vs.read()

  # Si ya no hay más frame, significa que el video termino y por tanto se sale del bucle
  if frame is None:
    break
  
  status = "Waiting"
  rects = []

  date_time = datetime.datetime.now()
  # Nos saltamos los frames especificados.
  if totalFrame % skip_fps == 0:
    status = "Detecting"
    trackers = []
    # Tomamos la imagen la convertimos a array luego a tensor
    image_np = np.array(frame)

    # pimage = process_image_yolo(image_np)

    # Predecimos los objectos y clases de la imagen
    boxes, classes, scores = yolo.predict(image_np)

    # transorm boxes

    # Recorremos las detecciones
    for x in range(len(classes)):
      idx = int(classes[x])
      # Tomamos los bounding box 
      (startX, startY, endX, endY) = np.array(boxes[x]).astype("int")

      # Con la función de dlib empezamos a hacer seguimiento de los boudiung box obtenidos
      tracker = dlib.correlation_tracker()
      rect = dlib.rectangle(startX, startY, endX, endY)
      tracker.start_track(frame, rect)

      trackers.append(tracker)
  else:
    # En caso de que no hagamos detección haremos seguimiento
    # Recorremos los objetos que se les está realizando seguimiento
    for tracker in trackers:
      status = "Tracking"
      # Actualizamos y buscamos los nuevos bounding box
      tracker.update(frame)
      pos = tracker.get_position()

      startX = int(pos.left())
      startY = int(pos.top())
      endX = int(pos.right())
      endY = int(pos.bottom())

      rects.append((startX, startY, endX, endY))

  # Dibujamos el umbral de conteo
  #cv2.rectangle(frame, (POINT[0], POINT[1]), (POINT[0]+ POINT[2], POINT[1] + POINT[3]), (255, 0, 255), 2)

  objects = ct.update(rects)
  objects_to_save = {}
  # Recorremos cada una de las detecciones
  for (objectID, centroid) in objects.items():
    # Revisamos si el objeto ya se ha contado
    to = trackableObjects.get(objectID, None)
    if to is None:
      to = trackableobject.TrackableObject(objectID, centroid)

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
    objects_to_save[objectID] = centroid.tolist()
    
    # Dibujamos el centroide y el ID de la detección encontrada
    text = "ID {}".format(objectID)
    cv2.putText(frame, text, (centroid[0]-10, centroid[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    cv2.circle(frame, (centroid[0], centroid[1]), 4, (0,255,0), -1)

  if objects_to_save:
    raw_data[str(date_time)] = objects_to_save

  # Totalizamos los resultados finales
  info = [
          ("Sur", totalUp),
          ("Norte", totalDown),
          ("Estado", status),
  ]

  # for (i, (k,v)) in enumerate(info):
  #   text = "{}: {}".format(k,v)
  #   cv2.putText(frame, text, (10, H - ((i*20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

  # Almacenamos el framme en nuestro video resultante.
  writer.write(frame)
  totalFrame += 1
  fps.update()

# Terminamos de analizar FPS y mostramos resultados finales
fps.stop()

print("Tiempo completo {}".format(fps.elapsed()))
print("Tiempo aproximado por frame {}".format(fps.fps()))

# Cerramos el stream the almacenar video y de consumir el video.
writer.release()
vs.release()

output_data = {
  "camera_location": camera_location,
  "period_time": period_time,
  "raw_data": raw_data
}
with open(DATA_DIR / 'data.json', 'w') as f:
    json.dump(output_data, f)