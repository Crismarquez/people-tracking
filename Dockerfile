FROM python:3.8
WORKDIR /app
COPY config/ ./config
COPY otracking/ ./otracking
COPY store/ ./store
COPY strong_sort/ ./strong_sort
COPY app.py .
COPY requirements.txt .
RUN apt-get update && yes | apt-get upgrade

# RUN apt-get install -y git python3-pip
RUN pip install --upgrade pip
RUN apt-get install libxmu6 libxmuu1 xauth xclip xsel -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install cmake cmake-data libarchive13 libjsoncpp24 librhash0 libuv1 -y
# RUN apt-get install build-essential cmake pkg-config
RUN pip install dlib
RUN pip install -r requirements.txt
RUN export PYTHONPATH="${PYTHONPATH}:${PWD}"
EXPOSE 8000
ENV PORT 8000
ENV HOST "0.0.0.0"
CMD ["python3", "./app.py"]
