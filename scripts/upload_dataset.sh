#!/bin/bash

# change to the dir of the script
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# change to the dir to the project
cd ..

title() {
    sharps="#################################"
    printf "\n%s\n%s\n%s\n" ${sharps} "$1" ${sharps}
}

upload_dataset_config=AI/upload_dataset.yaml

for name in \
    SX_spcas9 \
    SX_spymac \
    SX_ispymac

    title "Upload dataset ${name}"
    ./run.py upload_dataset \
        --config ${upload_dataset_config} \
        --dataset.config_name ${name}
