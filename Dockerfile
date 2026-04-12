FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y
RUN apt-get update --fix-missing -y

RUN apt-get install -y ffmpeg libsm6 libxrender1 libxtst6 zip 

RUN apt-get install -y p7zip-full

# Library components for av
RUN apt-get install -y \
    libavformat-dev libavcodec-dev libavdevice-dev \
    libavutil-dev libswscale-dev libswresample-dev libavfilter-dev


RUN apt-get install -y python3 python3-pip git python3-dev pkg-config htop wget

RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision
RUN pip3 install numpy pandas openpyxl scikit-image scikit-learn scipy opencv-python
RUN pip3 install matplotlib seaborn pydensecrf
RUN pip3 install wandb 
RUN pip3 install transformers diffusers huggingface_hub[cli]

WORKDIR /app/script
