FROM ros:indigo
MAINTAINER Leon Sixt

RUN rosdep update
RUN apt-get update --yes && apt-get upgrade --yes
RUN apt-get install --yes ros-indigo-desktop-full \
    liburdfdom-tools \
    openssh-server \
    vim \
    tmux \
    zsh \
    ros-indigo-ompl-visual-tools \
    ros-indigo-moveit-ros-* \
    ros-indigo-ros-control \
    ros-indigo-ros-controllers \
    ros-indigo-fake-localization

RUN apt-get install --yes \
    python-scipy \
    python-matplotlib \
    python-pip \
    python-flake8 \
    ipython \
    ipython-notebook


RUN echo "source /opt/ros/indigo/setup.bash" >> ~/.bashrc \
    && echo "AuthorizedKeysFile  %h/.ssh/authorized_keys\nX11UseLocalhost no" >> /etc/ssh/sshd_config

COPY authorized_keys /root/.ssh/authorized_keys
RUN mkdir -p /var/run/sshd
EXPOSE 22


# Set the locale
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

CMD ["/usr/sbin/sshd", "-D"]