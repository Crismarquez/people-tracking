"""
    Yolo models (v3, v5)
"""
from pathlib import Path
import cv2
import numpy as np
import torch

from otracking.utils import relativebox2absolutebox, download_models
from config.config import MODELS_DIR, YV5_FORMATS

ALLOW_DETECTOR_MODELS = ["yolov3"]
ALLOW_DETECTOR_MODELS_YV5 = ["yv5_onnx", "yv5_pt"]


class Yolov5:
    def __init__(self, model_name, confidence: float=0.6):

        self.model_name = model_name
        self.model_file = YV5_FORMATS[self.model_name]
        self.model_dir = Path(MODELS_DIR, self.model_name, self.model_file)

        self.confidence = confidence

        self.load_model()

    def load_model(self):
        if not self.model_dir.exists():
            print("model not exist in project directory, trying download")
            download_models(self.model_name)

        self.model = torch.hub.load(
            'ultralytics/yolov5', 'custom', path=str(self.model_dir), force_reload=True
        )
        self.model.conf = self.confidence

    def _process_output(self):
        pass

    def _predict(self, img: np.ndarray):
        detections = self.model(img)
        detections_crop = detections.crop(save=False)

        return detections_crop, detections

    def predict(self, img: np.ndarray):
        detections_crop, results = self._predict(img) 

        return detections_crop, results


class YOLO:
    def __init__(
        self,
        model_name:str = "yolov3",
        obj_threshold: float=0.7, 
        nms_threshold:float=0.4,
        filter_class: list=[0]):
        """Init.

        # Arguments
            obj_threshold: Integer, threshold for object.
            nms_threshold: Integer, threshold for box.
        """
        self.obj_threshold = obj_threshold
        self.nms_threshold = nms_threshold
        self.filter_class = filter_class
        self.model_name = model_name
        self.CONFIG_NAME = "yolov3.cfg"
        self.WEIGHTS_NAME = "yolov3.weights"

        if model_name not in ALLOW_DETECTOR_MODELS:
            raise ValueError(
                f"model {model_name} is not implemented, try someone: {ALLOW_DETECTOR_MODELS}"
                )
        
        self.model_dir = Path(MODELS_DIR, model_name)

        self.load_model()

    def load_model(self):
        if not self.model_dir.exists():
            print("model not exist in project directory, trying download")
            download_models(self.model_name)
        
        config_path = self.model_dir / self.CONFIG_NAME
        weights_path = self.model_dir / self.WEIGHTS_NAME
        self._yolo = cv2.dnn.readNetFromDarknet(str(config_path), str(weights_path))

    def img_preprocess(self, image):
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        return blob


    def _supress_boxes(self, boxes, confidences, class_ids):
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.obj_threshold, self.nms_threshold)

        boxes_filtered = []
        confidences_filtered = []
        class_ids_filtered = []

        if len(indexes) > 0:
            # loop over the indexes we are keeping
            for i in indexes.flatten():
                # extract only indexes supress
                boxes_filtered.append(boxes[i])
                confidences_filtered.append(confidences[i])
                class_ids_filtered.append(class_ids[i])

        return boxes_filtered, confidences_filtered, class_ids_filtered

    def _process_output(self, layer_output, shape):
        width, height = shape[1], shape[0]
        image_dims = [width, height, width, height]
        # detection 4 + 1 +80), output feature map of yolo.
        boxes = []
        confidences = []
        class_ids = []

        for output in layer_output:
            for detection in output:
                score = detection[5:]
                class_id = np.argmax(score)
                if class_id not in self.filter_class:
                    continue

                confidence = score[class_id]

                if confidence > self.obj_threshold:
                    box = detection[0:4] * np.array(image_dims)
                    (centerX, centerY, width, height) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        boxes, scores, classes = self._supress_boxes(boxes, confidences, class_ids)

        boxes = [relativebox2absolutebox(box) for box in boxes]

        return boxes, classes, scores


    def _predict(self, pimage):
        self._yolo.setInput(pimage)
        output_layers_name = self._yolo.getUnconnectedOutLayersNames()
        layer_output = self._yolo.forward(output_layers_name)

        return layer_output


    def predict(self, image):
        """Detect the objects with yolo.

        # Arguments
            image: ndarray, input image.

        # Returns
            boxes: List, boxes of objects.
            classes: List, classes of objects.
            scores: List, scores of objects.
        """

        shape = image.shape
        pimage = self.img_preprocess(image)
        layer_output = self._predict(pimage)
        boxes, classes, scores = self._process_output(layer_output, shape)

        return boxes, classes, scores
