#!/bin/bash

BUILDID="build-$RANDOM"
INSTANCE="travisci/ci-ubuntu-2004:packer-minimal-1644996667-31a09d16"
docker run -t -d --name $BUILDID -it $INSTANCE bash
docker exec -it $BUILDID bash -l
