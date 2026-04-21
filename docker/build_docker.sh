#!/bin/bash

# change to the dir of the script
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# change to the dir to the project
cd ..

cat .gitignore docker/dockerignoretail > .dockerignore

docker build . \
    -f docker/Dockerfile \
    -t ghcr.io/ljw20180420/crispr_ai:latest
