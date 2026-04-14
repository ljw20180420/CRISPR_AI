#!/bin/bash

# change to the dir of the script
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# change to the dir to the project
cd ../..

function title() {
    sharps="#################################"
    printf "\n%s\n%s\n%s\n" ${sharps} $1 ${sharps}
}

app_config=AI/app.yaml
target="GreatestCommonCrossEntropy"
maximize_target="false"
device=${device:-"cuda"}
autocast=${autocast:-"false"}
owner="ljw20180420"

printf "inference:\n" > ${app_config}
for data_name in SX_spcas9 SX_spymac SX_ispymac
do
    printf "  - inference/inference.yaml\n" >> ${app_config}
done

printf "\ntest:\n" >> ${app_config}
for data_name in SX_spcas9 SX_spymac SX_ispymac
do
    printf "  - checkpoints_path: %s/CRIfuser_CRIfuser_%s\n" \
        ${owner} ${data_name} \
        >> ${app_config}
    printf "    logs_path: %s/CRIfuser_CRIfuser_%s\n" \
        ${owner} ${data_name} \
        >> ${app_config}
    printf "    target: %s\n" \
        ${target} \
        >> ${app_config}
    printf "    maximize_target: %s\n" \
        ${maximize_target} \
        >> ${app_config}
    printf "    overwrite:\n      train.device: %s\n      model.init_args.autocast: %s\n" \
        ${device} ${autocast} \
        >> ${app_config}
done

./run.py app --config ${app_config}
