#!/bin/bash

# change to the dir of the script
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# change to the dir to the project
cd ..

cat .gitignore docker/dockerignoretail > .dockerignore

mkdir -p docker/genome
if ! [ -f "docker/genome/hg19.2bit" ]
then
    ln "${GENOME}" "docker/genome/hg19.2bit"
fi
for bt2 in 1.bt2 2.bt2 3.bt2 4.bt2 rev.1.bt2 rev.2.bt2
do
    if ! [ -f "docker/genome/hg19.${bt2}" ]
    then
        ln "${BOWTIE2_INDEX}.${bt2}" "docker/genome/hg19.${bt2}"
    fi
done

docker build . \
    -f docker/Dockerfile \
    -t ghcr.io/ljw20180420/crispr_ai:latest
