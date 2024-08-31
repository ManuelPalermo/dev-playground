
# After build contains:
# - ubuntu22.04
# - nvidia-cuda12.6
# - cpp tools
# - python tools
# - shell tools
# - xauth/X11
# - miniconda

FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

ARG USER=user
ENV UID=1000
ENV GID=${UID}
ENV HOME=/home/${USER}
ENV WORKSPACE_NAME="dev-playground"

ENV DEBIAN_FRONTEND=noninteractive

RUN groupadd -g ${GID} ${USER} \
    && useradd -l -s /bin/bash -u ${UID} -g ${USER} -m -d ${HOME} ${USER} \
    && apt-get update && apt-get install -y sudo \
    && echo ${USER} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USER} \
    && chmod 0440 /etc/sudoers.d/${USER} \
    && rm -rf /var/lib/apt/lists/* && rm -rf /var/cache/apt && apt-get clean

# install language
RUN sudo apt-get update && sudo apt-get install -y \
    locales \
    && sudo locale-gen en_US.UTF-8 \
    && sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && sudo rm -rf /var/lib/apt/lists/* && sudo  rm -rf /var/cache/apt && sudo apt-get clean
ENV LANG=en_US.UTF-8

# install timezone
RUN sudo ln -fs /usr/share/zoneinfo/UTC /etc/localtime \
    && export DEBIAN_FRONTEND=noninteractive \
    && sudo apt-get update \
    && sudo apt-get install -y tzdata \
    && sudo dpkg-reconfigure --frontend noninteractive tzdata \
    && sudo rm -rf /var/lib/apt/lists/* && sudo rm -rf /var/cache/apt && sudo apt-get clean

# install dev tools
RUN sudo apt-get update && sudo apt-get install -y \
    apt-utils \
    gnupg2 \
    lsb-release \
    gdb \
    git \
    curl \
    wget \
    tree \
    iproute2 \
    htop \
    nano \
    vim \
    unzip \
    ffmpeg \
    build-essential \
    python3-pip \
    pre-commit \
    qt6-base-dev \
    && sudo rm -rf /var/lib/apt/lists/* && sudo rm -rf /var/cache/apt && sudo apt-get clean

# update installed libs
RUN sudo apt-get update && sudo apt-get upgrade -y \
    && sudo rm -rf /var/lib/apt/lists/* && sudo rm -rf /var/cache/apt && sudo apt-get clean

# install cpp code compiler/checking/formatting tools
RUN sudo apt-get update && sudo apt-get install -y \
    cmake \
    ninja-build \
    clang \
    clang-format \
    clang-tidy \
    cppcheck \
    cpplint \
    doxygen \
    libgtest-dev \
    libstdc++-10-dev \
    protobuf-compiler \
    && sudo rm -rf /var/lib/apt/lists/* && sudo rm -rf /var/cache/apt && sudo apt-get clean

# install python code checking/formatting/testing tools
RUN pip3 install \
    ruff \
    pyupgrade \
    mypy \
    sphinx \
    && sudo rm -rf /var/lib/apt/lists/* && sudo rm -rf /var/cache/apt && sudo apt-get clean

# install shell code checking/formatting tools
RUN sudo apt-get update && sudo apt-get install -y \
    shfmt \
    shellcheck \
    && sudo rm -rf /var/lib/apt/lists/* && sudo rm -rf /var/cache/apt && sudo apt-get clean

# add GUI support using X11
RUN sudo apt-get update && sudo apt-get install -y \
    xauth \
    libgl1-mesa-glx \
    && sudo rm -rf /var/lib/apt/lists/* && sudo rm -rf /var/cache/apt && sudo apt-get clean

# install oh-my-bash for a better shell
RUN bash -c "$(curl -fsSL https://raw.githubusercontent.com/ohmybash/oh-my-bash/master/tools/install.sh)" --unattended

# install miniconda for DL related tasks
RUN mkdir -p ~/miniconda3/ \
    && wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O ~/miniconda3/Miniconda3-latest-Linux-x86_64.sh \
    && bash ~/miniconda3/Miniconda3-latest-Linux-x86_64.sh -b -u


ENV DEBIAN_FRONTEND=


USER ${USER}
RUN mkdir -p /home/${USER}/${WORKSPACE_NAME}
