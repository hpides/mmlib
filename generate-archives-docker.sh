#!/bin/bash

cd "$(dirname "$0")"

CONTAINER_NAME=mmlib-python

docker run --rm --name $CONTAINER_NAME -it -d python:3.8
docker cp ../mmlib $CONTAINER_NAME:/
docker exec $CONTAINER_NAME /mmlib/generate-archives.sh
docker cp $CONTAINER_NAME:/mmlib/dist ./
docker kill $CONTAINER_NAME
