# Nvidia Docker for ros indigo

This folder includes a Dockerfile that includes ros indigo and cudnn.
To build the base image, clone this repo: https://github.com/berleon/ros_cudnn_docker_images
And build the images with

```bash
$ git clone https://github.com/berleon/ros_cudnn_docker_images.git
$ cd ros_cudnn_docker_images/ros/indigo/
$ make build_desktop
```

Make sure all commands are run within the docker directory!
```bash
# to build the docker file. This will automatically append your public ssh key
# to the authorized_keys file
$ ./build.sh

# create your own user inside of docker
$ ./docker_userify.sh

$ cd ${USER}_deep_car

# build your image.
$ ./build.sh

# run your docker image.
$ ./run.sh
```

You can even forward an X session over ssh:

```
$ ssh -X -Y yourname@yourhost.com
```
