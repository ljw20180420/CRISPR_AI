#!/bin/bash

# change to the dir of the script
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# change to the dir to the project
cd ../..

function title() {
    sharps="#################################"
    printf "\n%s\n%s\n%s\n" ${sharps} $1 ${sharps}
}

train_config=AI/train.yaml
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
        model_config=AI/preprocess/${preprocess}/${model_cls}.yaml

        title Train/Eval
        case ${model_cls} in
            XGBoost)
                ./run.py train --config ${train_config} --train.output_dir ${output_dir} --train.trial_name default --train.device cpu --train.evaluation_only true --dataset.name ${data_name} --model ${model_config}
            ;;
            *)
                ./run.py train --config ${train_config} --train.output_dir ${output_dir} --train.trial_name default --train.evaluation_only true --dataset.name ${data_name} --model ${model_config}
            ;;
        esac
    done
done