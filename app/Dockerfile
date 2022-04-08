FROM tensorflow/tensorflow:2.7.1

LABEL org.opencontainers.image.authors="johns@provoke.co.nz"

COPY . /tf

WORKDIR /tf

RUN apt-get update && \
    pip install --upgrade pip \
    pip --no-cache install \
    mlflow --quiet \
    matplotlib \
    && apt-get install wget -y \
    && wget https://dvc.org/deb/dvc.list -O /etc/apt/sources.list.d/dvc.list \
    && apt-get update \
    && apt-get install dvc -y

ENTRYPOINT ["./runit.sh"]
