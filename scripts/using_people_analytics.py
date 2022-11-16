import cv2
from pathlib import Path

from otracking.models import PeopleAnalytics
from config.config import DATA_DIR

name_file = "1_pasillo_mall_large.mp4"
path_video = DATA_DIR / name_file
path_out = DATA_DIR / ("out_" + name_file)

camera_location = "pasillo_2"
period_time = "2022-08-05-10am_2022-08-05-11am"
relative_mask = [[0.3, 0.42],
 [0.74, 0.42],
 [0.74, 0.51],
 [1.0, 0.83],
 [1.0, 1.0],
 [0.0, 1.0],
 [0.0, 0.74]]

skip_fps = 30
threshold = 0.3

contents = cv2.VideoCapture(str(path_video))

with open(path_video, "rb") as file:
    contents = file.read()

model = PeopleAnalytics(camera_location, period_time, detector_name="yv5_onnx")
response = model.process_video(contents, region_mask=relative_mask, draw_video=True)

video_bs64 = response["draw_video"]
