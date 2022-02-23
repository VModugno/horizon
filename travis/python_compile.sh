#!/bin/bash

FOREST_ARGS="--default-build-type Release --clone-protocol https --verbose -j2"
cd $FOREST_DIR
forest casadi $FOREST_ARGS
forest pybind11 $FOREST_ARGS
cd $HORIZON_DIR
pip3 install .