FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04

# Hack to not have tzdata cmdline config during build
RUN ln -fs /usr/share/zoneinfo/Europe/Amsterdam /etc/localtime
# Install python3.7 and dependencies, taken from:
# - hhttps://websiteforstudents.com/installing-the-latest-python-3-7-on-ubuntu-16-04-18-04/
# - https://github.com/zju3dv/pvnet/blob/master/docker/Dockerfile
# - https://github.com/zju3dv/clean-pvnet
RUN apt-get update && \
    apt install software-properties-common -yq
RUN add-apt-repository ppa:deadsnakes/ppa
RUN add-apt-repository ppa:rock-core/qt4

RUN apt update && \
    apt install -yq software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install -yq \
        nano \
        sudo \
        wget \
        curl \
        build-essential \
        cmake \
        git \
        ca-certificates \
        python3.7 \
        python3-pip \
        libjpeg-dev \
        zip \
        unzip \
        libpng-dev \
        libeigen3-dev \
        libglfw3-dev \
        libglfw3 \
        libgoogle-glog-dev \
        libsuitesparse-dev \
        libatlas-base-dev  \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
# (mini)conda
# https://repo.anaconda.com/miniconda/
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh && \
    sh ./Miniconda3-py37_4.8.3-Linux-x86_64.sh -b -p /opt/conda && \
    rm ./Miniconda3-py37_4.8.3-Linux-x86_64.sh && \
    export PATH=$PATH:/opt/conda/bin && \
    conda install conda-build

ENV PATH $PATH:/opt/conda/envs/env/bin:/opt/conda/bin
RUN add-apt-repository universe  && apt update 
RUN apt-get install gcc-7 g++-7 libsm6 libxrender1 libfontconfig1 -y
RUN rm /usr/bin/gcc /usr/bin/g++ usr/bin/cc /usr/bin/c++
RUN ln -s /usr/bin/gcc-7 /usr/bin/gcc
RUN ln -s /usr/bin/g++-7 /usr/bin/g++
RUN ln -s /usr/bin/gcc-7 /usr/bin/cc
RUN ln -s /usr/bin/g++-7 /usr/bin/c++
RUN cd /opt && \
    conda init bash && \
    conda create -n pvnet python=3.8 && \
    . /opt/conda/etc/profile.d/conda.sh && \
    conda activate pvnet && \
    conda install pyqt && \
    pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116 && \
    pip install --user Cython==0.28.2 yacs open3d-python==0.3.0.0 opencv-python pycocotools plyfile && \
    pip install --user Pillow==6.2.1 imagezmq && \
    pip install --user yacs==0.1.4 numpy==1.18.0 opencv-python tqdm==4.28.1 pycocotools==2.0.0 matplotlib==2.2.2  && \
    pip install --user plyfile==0.6 scikit-image==0.14.2 scikit-learn PyOpenGL==3.1.1a1  cyglfw3==3.1.0.2  && \
    pip install --user pyassimp==3.3 progressbar==2.5 tensorboardX==1.2  && \ 
    pip install --user transforms3d ipdb cffi open3d
    
#ipdb==0.11 cffi==1.11.5
CMD ["/bin/bash"]
