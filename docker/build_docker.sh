#!/bin/bash

# change to the dir of the script
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# change to the dir to the project
cd ..

cat .gitignore docker/dockerignoretail > .dockerignore
rm -r docker/common_ai
mkdir -p docker/common_ai/common_ai
rsync -av --exclude='__pycache__' --exclude="*.yaml" ${PYTHONPATH}/common_ai/ docker/common_ai/common_ai/

docker build . \
    -f docker/Dockerfile \
    -t ghcr.io/ljw20180420/crispr_ai:latest
