#!/bin/bash
set -ev

sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo apt update && sudo apt install libgtest-dev libyaml-cpp-dev python3-pip python3-setuptools liburdfdom-dev cmake libeigen3-dev libboost-filesystem-dev libboost-serialization-dev libblas-dev liblapack-dev patchelf python3-venv python3-wheel


if [ "$DISTRIBUTION" = "python" ]; then
    # prepare environment for python install
    ./travis/forest_init.sh;
fi


if [ "$DISTRIBUTION" = "conda" ]; then
    # prepare environment for python install
    source ./travis/conda_init.sh;
fi