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
device=${device:-"cuda"}
owner="ljw20180420"

printf "inference:\n" > ${app_config}
for data_name in SX_spcas9 SX_spymac SX_ispymac
do
    for pre_model in \
        CRIformer:CRIformer \
        inDelphi:inDelphi \
        Lindel:Lindel \
        DeepHF:DeepHF \
        DeepHF:CNN \
        DeepHF:MLP \
        DeepHF:XGBoost \
        DeepHF:SGDClassifier \
        CRIfuser:CRIfuser \
        FOREcasT:FOREcasT
    do
        printf "  - inference/inference.yaml\n" >> ${app_config}
    done
done

printf "\ntest:\n" >> ${app_config}
for data_name in SX_spcas9 SX_spymac SX_ispymac
do
    for pre_model in \
        CRIformer:CRIformer \
        inDelphi:inDelphi \
        Lindel:Lindel \
        DeepHF:DeepHF \
        DeepHF:CNN \
        DeepHF:MLP \
        DeepHF:XGBoost \
        DeepHF:SGDClassifier \
        CRIfuser:CRIfuser \
        FOREcasT:FOREcasT
    do
        IFS=":" read preprocess model_cls <<<${pre_model}

        printf "  - checkpoints_path: %s/%s_%s_%s\n" \
            ${owner} ${preprocess} ${model_cls} ${data_name} \
            >> ${app_config}
        printf "    logs_path: %s/%s_%s_%s\n" \
            ${owner} ${preprocess} ${model_cls} ${data_name} \
            >> ${app_config}
        printf "    target: %s\n" \
            ${target} \
            >> ${app_config}
        printf "    overwrite:\n      train.device: %s\n" \
            ${device} \
            >> ${app_config}
    done
done

./run.py app --config ${app_config}
