FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime

RUN pip install pprintpp opencv-python tensorboard torchmetrics

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt update
RUN apt install vim -y
RUN pip install pycocotools
COPY ./src /workspace/src
COPY ./demo /workspace/demo


WORKDIR /workspace/src
CMD ["python", "train.py"]
