import io
from typing import List
from fastapi import FastAPI, UploadFile, Query
import uvicorn

from otracking.models import PeopleAnalytics

app = FastAPI()



@app.get('/')
async def read_root():
    return {"message": "Api People Analitycs - BlueLabs"}


@app.post('/process-video', status_code=200)
async def parse_request(
    camera_location:str,
    period_time:str,
    file: UploadFile,
    draw_video=False 
    ):

    region_mask= [[0.3, 0.42],
                    [0.74, 0.42],
                    [0.74, 0.51],
                    [1.0, 0.83],
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [0.0, 0.74]]
     
    contents = await file.read()
    model = PeopleAnalytics(camera_location, period_time, detector_name="yv5_onnx")
    response = model.process_video(contents, region_mask=region_mask, draw_video=draw_video)

    return response["output_data"]


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)