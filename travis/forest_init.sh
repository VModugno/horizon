#!/bin/sh

sudo pip3 install build twine hhcm_forest==0.0.26
mkdir $BUILD_DIR
mkdir $FOREST_DIR
cd $FOREST_DIR
forest --init
echo "source $FOREST_DIR/setup.bash" >> ~/.bashrc && source ~/.bashrc
ln -s $SRC_DIR src/$(basename $SRC_DIR)  # symlink original source folder
forest -a git@github.com:advrhumanoids/multidof_recipes.git master -u --clone-protocol https  # get recipes