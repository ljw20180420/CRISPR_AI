#!/bin/bash

# change to the dir of the script
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# change to the dir to the project
cd ..

cp .gitignore .dockerignore
printf ".git\n.conda\ndocker\npaper\n.vscode\nscripts/formal/*\n!scripts/formal/app.sh\n" >> .dockerignore
printf "AI/dataset/*\n!AI/dataset/dataset.yaml\n!AI/dataset/test.json.gz\n!AI/dataset/inference.csv.gz\n" >> .dockerignore

docker build . \
    --build-arg HTTP_PROXY=http://10.0.2.2:1081 \
    --build-arg HTTPS_PROXY=http://10.0.2.2:1081 \
    -f docker/Dockerfile \
    -t ghcr.io/ljw20180420/crispr_ai:latest
