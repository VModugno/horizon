#!/bin/bash

REPO=francescoruscelli
./build.sh
docker tag horizon $REPO/horizon
docker login --username=$REPO --password=$DOCKER_TOKEN
docker push $REPO/horizon
