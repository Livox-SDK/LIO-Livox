# Docker for LIO-Livox

Using Docker is a quick and easy wasy to run LIO-Livox.

## 1. Run the prebuilt dockers

### Run the docker environment

- Install Docker: <https://docs.docker.com/engine/install/>.

- Pull and run the Docker container: For the Melodic-based image, type following command to pull the Docker image and run a container.

```
docker run -d --network host --ipc host -e DISPLAY --volume /tmp/.X11-unix:/tmp/.X11-unix --volume /dev:/dev --privileged vietanhdev/lio_livox:ros-melodic-lio-1.0
```

See the output or type "docker ps" to see the docker container id. Open a new terminal into the container by:

```
docker exec -it <container-id> bash
```

For example:

```
➜  ~ docker run -d --network host --ipc host -e DISPLAY --volume /tmp/.X11-unix:/tmp/.X11-unix --volume /dev:/dev --privileged vietanhdev/lio_livox:ros-melodic-lio-1.0
eeb215f0c25e
➜  ~ docker exec -it eeb215f0c25e bash
root@visionpc:/home/ros/catkin_ws# 
```

- Open LIO-Livox in a Docker container. Replace `eeb215f0c25e` with your container id:

```
docker exec -it eeb215f0c25e bash
roslaunch lio_livox horizon.launch
```

You should see a window opened. If not, try running `xhost +local:docker` first and try again.

- Open another window in the host machine and play the bag file:

```
rosbag play YOUR_ROSBAG.bag
```

**Note:** You can replace "melodic" with "kinetic" in all the above commands to run the Kinetic-based Docker image. I tested running ROS Kinetic in the Docker container and ROS Melodic on the host machine without any problem. You can still send and receive ROS messages between the inside and outside nodes.

## 2. Rebuild the dockers

- Build a docker from ROS Melodic base:

```
docker build . -f Dockerfile-melodic -t <tag-name>
```

- Build a docker from ROS Kinetic base:

```
docker build . -f Dockerfile-melodic -t <tag-name>
```