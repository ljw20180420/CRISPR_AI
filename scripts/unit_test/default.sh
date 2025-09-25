#!/bin/bash

# change to the dir of the script
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# change to the dir to the project
cd ../..

train_config=AI/preprocess/train.yaml
output_dir=${OUTPUT_DIR:-$HOME"/CRISPR_results"}/unit_test/default
test_config=AI/preprocess/test.yaml

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
        IFS=":" read preprocess model_type <<<${pre_model}
        model_config=AI/preprocess/${preprocess}/${model_type}.yaml

        # Train
        ./run.py train --config ${train_config} --train.output_dir ${output_dir} --train.trial_name default --train.num_epochs 1 --dataset.data_file AI/dataset/test.json.gz --dataset.name ${data_name} --model ${model_config}

        # Eval
        ./run.py train --config ${train_config} --train.output_dir ${output_dir} --train.trial_name default --train.num_epochs 1 --train.evaluation_only true --dataset.data_file AI/dataset/test.json.gz --dataset.name ${data_name} --model ${model_config}

        # Test
        checkpoints_path=${output_dir}/checkpoints/${preprocess}/${model_type}/${data_name}/default
        logs_path=${output_dir}/logs/${preprocess}/${model_type}/${data_name}/default
        for target in \
            CrossEntropy \
            NonZeroCrossEntropy \
            NonWildTypeCrossEntropy \
            NonZeroNonWildTypeCrossEntropy \
            GreatestCommonCrossEntropy
        do
            ./run.py test --config ${test_config} --test.checkpoints_path ${checkpoints_path} --test.logs_path ${logs_path} --test.target ${target}
        done
    done
done