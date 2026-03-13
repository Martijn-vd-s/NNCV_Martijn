FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*


# create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"


# install dependencies (add any additional dependencies your project requires)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu126
RUN pip install --no-cache-dir \
    pillow\
    segmentation_models_pytorch \
    torchmetrics \
    termcolor


# Create the PyTorch cache directory inside the container
RUN mkdir -p /root/.cache/torch/hub/checkpoints/



WORKDIR /app
COPY predict.py /app/predict.py
COPY model.py /app/model.py
COPY model.pt /app/model.pt

COPY facebookresearch_dinov3_main /app/facebookresearch_dinov3_main
COPY dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth /app/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth

ENTRYPOINT ["python", "/app/predict.py"]