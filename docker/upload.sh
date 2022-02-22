#!/bin/bash

REPO=francescoruscelli
./docker/build.sh
docker tag horizon $REPO/horizon
docker login --username=$REPO --password=$DOCKER_TOKEN
docker push $REPO/horizon
