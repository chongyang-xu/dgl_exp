FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04 AS builder

WORKDIR /workspace
VOLUME /workspace
VOLUME /data

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git	\
    vim \
    cmake \
    g++ \
    gcc \
    make \
    build-essential \
    python3-dev \
    python3-pip \
    iproute2 \
    inetutils-ping


# ssh setup
RUN apt install -y openssh-server
RUN sed -ri 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -ri 's/^UsePAM yes/UsePAM no/' /etc/ssh/sshd_config
RUN mkdir /root/.ssh/
RUN mkdir /run/sshd
COPY ssh_cfg/id_rsa /root/.ssh/
COPY ssh_cfg/authorized_keys /root/.ssh/
RUN chmod 0400 /root/.ssh/id_rsa
RUN chmod 0400 /root/.ssh/authorized_keys

#install pre-built pytorch
RUN pip3 install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install ogb==1.3.6 scipy==1.10.1 networkx==2.1 psutil==5.8.0 pyyaml pandas

#CMD ["/usr/sbin/sshd", "-D"]
#CMD ["/bin/bash"]

