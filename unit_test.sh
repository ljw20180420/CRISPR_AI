#!/bin/bash

# change to the dir of the script
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

train_config=AI/preprocess/train.yaml
output_dir=${OUTPUT_DIR:-$HOME}
test_config=AI/preprocess/test.yaml

for data_name in SX_spcas9
# for data_name in SX_spcas9 SX_spymac SX_ispymac
do
    for pre_model in DeepHF:DeepHF
    # for pre_model in \
    #     CRIformer:CRIformer \
    #     inDelphi:inDelphi \
    #     Lindel:Lindel \
    #     DeepHF:DeepHF \
    #     DeepHF:CNN \
    #     DeepHF:MLP \
    #     DeepHF:XGBoost \
    #     DeepHF:Ridge \
    #     CRIfuser:CRIfuser \
    #     FOREcasT:FOREcasT
    do
        IFS=":" read preprocess model_type <<<${pre_model}
        model_config=AI/preprocess/${preprocess}/${model_type}.yaml
        # Train
        ./run.py train --config ${train_config} --train.output_dir ${output_dir} --train.trial_name unit_test --train.num_epochs 1 --dataset.data_file AI/dataset/test.json.gz --dataset.name ${data_name} --model ${model_config}

        model_path=${output_dir}/${preprocess}/${model_type}/${data_name}/unit_test
        # Test
        ./run.py test --config ${test_config} --test.model_path ${model_path} --test.target CrossEntropy
        ./run.py test --config ${test_config} --test.model_path ${model_path} --test.target NonZeroCrossEntropy
        ./run.py test --config ${test_config} --test.model_path ${model_path} --test.target NonWildTypeCrossEntropy
        ./run.py test --config ${test_config} --test.model_path ${model_path} --test.target NonZeroNonWildTypeCrossEntropy
    done
done