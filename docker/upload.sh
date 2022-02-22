#!/bin/bash

REPO=francescoruscelli
cd docker && ./build
docker tag horizon $REPO/horizon
docker login --username=$REPO --password=$DOCKER_TOKEN
docker push $REPO/horizon;
