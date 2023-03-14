FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
WORKDIR /app
COPY ./requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt