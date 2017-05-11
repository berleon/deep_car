FROM ros_cudnn:indigo-desktop-full
MAINTAINER Leon Sixt

RUN rosdep update
RUN apt-get update --yes && apt-get upgrade --yes
RUN apt-get install --yes ros-indigo-desktop-full \
    liburdfdom-tools \
    openssh-server \
    ssl-cert \
    vim \
    tmux \
    zsh \
    python-setuptools \
    python-pip \
    python-nose \
    python-pytest \
    python-sklearn \
    python-skimage \
    python-h5py \
    python-matplotlib \
    python-scipy \
    python-matplotlib \
    python-pip \
    python-flake8 \
    ipython \
    ipython-notebook

RUN pip install --upgrade pip

RUN pip install \
    tensorflow-gpu \
    keras \
    jupyter \
    xgboost \
    seaborn \
    pytest-cov \
    pytest-benchmark \
    pytest-flake8 \
    more_itertools \
    scikit-image \
    click \
    pandas

RUN ldconfig    # otherwise the cuda libaries are not found

# Set the locale
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8