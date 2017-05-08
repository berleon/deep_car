#! /usr/bin/env bash

PORT=22222
CIDFILE="cid"

if [ -e $CIDFILE ]; then
    CID=$(cat $CIDFILE)
    CMD="docker start $CID"
    $CMD
else
    CMD="docker run --privileged=true \
        --cidfile=$CIDFILE \
        --env="DISPLAY"\
        -v /dev/dri:/dev/dri \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v /etc/machine-id:/etc/machine-id \
        -p 8888:8888 \
        -p $PORT:22 \
        -v $HOME:$HOME  \
        deep_car"
    echo "$CMD"
    $CMD
fi


