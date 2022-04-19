FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

RUN apt-key del 7fa2af80
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN rm /etc/apt/sources.list.d/cuda.list
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update && apt-get upgrade -y

RUN apt-get install -y curl git python3-pip vim wget

RUN apt-get install -y libjpeg-dev libopenmpi-dev zlib1g-dev

RUN pip3 install torch==1.6+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install numpy==1.16 imageio mpi4py sklearn
RUN git clone https://github.com/NVIDIA/apex
RUN cd apex && pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
