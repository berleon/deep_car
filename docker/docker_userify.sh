#! /usr/bin/env bash

if [ "$1" == "--help" ]; then
    echo "$0 generate user for the docker file"
    echo "    -u | --user NAME          name of the user. default: current user"
    echo "    --uid UID                 uid of the user. default: uid of current user"
    echo "    -p | --password PASSWORD  password of the user. default: username"
    exit 0
fi

while [[ $# -gt 1 ]]
do
key="$1"

case $key in
    -u|--user)
    USER="$2"
    shift # past argument
    ;;
    --uid)
    UID="$2"
    shift # past argument
    ;;
    -p|--password)
    PASSWORD="$2"
    shift # past argument
    ;;
    *)
            # unknown option
    ;;
esac
shift # past argument or value
done

echo $USER
echo $UID
echo $PASSWORD
if [ "$PASSWORD" == "" ]; then
    PASSWORD="$USER"
fi


OUT_DIR="${USER}_deep_car"
mkdir -p  "$OUT_DIR"

cat <<EOF  > $OUT_DIR/Dockerfile
FROM deep_car
MAINTAINER github@leon-sixt.de

RUN groupadd --gid $UID $USER
RUN useradd --uid $UID  --gid $USER \
    --home-dir /home/$USER --shell /usr/bin/zsh  \
    --groups sudo,$USER \
    --password $USER \
    $USER
# default password $USER
RUN echo $USER:$PASSWORD | chpasswd && \
    echo root:$PASSWORD | chpasswd

RUN chsh -s /usr/bin/zsh

USER $USER
WORKDIR /home/$USER
EOF

cat <<EOF > $OUT_DIR/build.sh
docker build -t ${USER}_deep_car .
EOF

chmod +x $OUT_DIR/build.sh

