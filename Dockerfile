FROM tensorflow/tensorflow:2.4.1-gpu

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libxext6 \
    libx11-6 \
  && rm -rf /var/lib/apt/lists/*

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris

RUN apt-get update \
  && apt-get install -y python-opengl python3-tk\
  && rm -rf /var/lib/apt/lists/*

RUN pip install numpy

RUN pip install matplotlib

RUN pip install gym

WORKDIR /gym

COPY ./src .
#CMD [ "python", "./random_policy.py" ]
CMD [ "python", "./run.py" ]