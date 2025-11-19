#!/bin/bash

# change to the dir of the script
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# change to the dir to the project
cd ../..

function title() {
    sharps="#################################"
    printf "\n%s\n%s\n%s\n" ${sharps} $1 ${sharps}
}

output_dir=${OUTPUT_DIR:-$HOME"/CRISPR_results"}/formal/default

for data_name in SX_spcas9 SX_spymac SX_ispymac
do
    title ${data_name}
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
        title ${pre_model}

        IFS=":" read preprocess model_cls <<<${pre_model}

        title "upload"
        hf upload \
            ${preprocess}_${model_cls}_${data_name} \
            ${output_dir}/checkpoints/${preprocess}/${model_cls}/${data_name}/default \
            checkpoints \
            --delete="*"

        hf upload \
            ${preprocess}_${model_cls}_${data_name} \
            ${output_dir}/logs/${preprocess}/${model_cls}/${data_name}/default/train \
            logs/train \
            --delete="*"
    done
done
