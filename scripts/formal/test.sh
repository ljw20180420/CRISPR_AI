#!/bin/bash

# change to the dir of the script
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# change to the dir to the project
cd ../..

function title() {
    sharps="#################################"
    printf "\n%s\n%s\n%s\n" ${sharps} $1 ${sharps}
}

test_config=AI/test.yaml
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
        checkpoints_path=${output_dir}/checkpoints/${preprocess}/${model_cls}/${data_name}/default
        logs_path=${output_dir}/logs/${preprocess}/${model_cls}/${data_name}/default

        title Test
        for target in \
            CrossEntropy \
            NonZeroCrossEntropy \
            NonWildTypeCrossEntropy \
            NonZeroNonWildTypeCrossEntropy \
            GreatestCommonCrossEntropy
        do
            ./run.py test --config ${test_config} --checkpoints_path ${checkpoints_path} --logs_path ${logs_path} --target ${target}
        done
    done
done
