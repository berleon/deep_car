# Docker for ros indigo

This folder includes a Dockerfile for ros indigo. As a special feature,
you can use ssh to connect to the container and forward the X-Server.

Make sure all commands are run within the docker directory!
```
# to build the docker file. This will automatically append your public ssh key
# to the authorized_keys file
$ ./build.sh

# run the docker image. This will also start the ssh deamon.
$ ./run.sh

# now you can connect to it via ssh
$ ./connect.sh

# if you shutdown, to restart the old container just type:
$ ./run.sh
```
