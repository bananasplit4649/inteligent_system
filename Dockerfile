FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04

COPY requirements.txt /root/

RUN apt-get update && apt-get install -y --no-install-recommends wget build-essential libreadline-dev \
libncursesw5-dev libssl-dev libsqlite3-dev libgdbm-dev libbz2-dev liblzma-dev zlib1g-dev uuid-dev libffi-dev libdb-dev \
swig git curl apt-utils make
RUN wget --no-check-certificate https://www.python.org/ftp/python/3.8.2/Python-3.8.2.tgz \
&& tar -xf Python-3.8.2.tgz \
&& cd Python-3.8.2 \
&& ./configure --enable-optimizations\
&& make \
&& make install
RUN pip3 install --upgrade pip
RUN pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install -r /root/requirements.txt

WORKDIR /work
