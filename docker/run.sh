#! /usr/bin/env bash

IMAGE=${USER}_deep_car
CMD="docker run -it \
    --privileged=true \
    --cidfile=cid \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --net=host \
    -v $HOME:$HOME  \
    --user="$UID" \
    --env="DISPLAY=$USER" \
    --env="USER=$USER" \
    --env="HOME=$HOME" \
    -w="$HOME" \
    $IMAGE
    zsh"
echo "$CMD"
$CMD


