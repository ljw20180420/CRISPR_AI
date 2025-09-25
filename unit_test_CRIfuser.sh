#!/bin/bash

# change to the dir of the script
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

train_config=AI/preprocess/train.yaml
output_dir=${OUTPUT_DIR:-$HOME}
test_config=AI/preprocess/test.yaml
loss_weight=1.0

for data_name in SX_spcas9 SX_spymac SX_ispymac
do
    for loss_function in \
        double_sample_negative_ELBO \
        importance_sample_negative_ELBO \
        forward_negative_ELBO \
        reverse_negative_ELBO \
        sample_CE non_sample_CE
    do
        # Train
        ./run.py train --config ${train_config} --train.output_dir ${output_dir} --train.trial_name ${loss_function}_test --train.num_epochs 1 --dataset.data_file AI/dataset/test.json.gz --dataset.name ${data_name} --model AI/preprocess/CRIfuser/CRIfuser.yaml --model.loss_weights "{'${loss_function}': ${loss_weight}}"

        # Eval
        ./run.py train --config ${train_config} --train.output_dir ${output_dir} --train.trial_name ${loss_function}_test --train.num_epochs 1 --train.evaluation_only true --dataset.data_file AI/dataset/test.json.gz --dataset.name ${data_name} --model AI/preprocess/CRIfuser/CRIfuser.yaml --model.loss_weights "{'${loss_function}': ${loss_weight}}"

        # Test
        checkpoints_path=${output_dir}/checkpoints/CRIfuser/CRIfuser/${data_name}/${loss_function}_test
        logs_path=${output_dir}/logs/CRIfuser/CRIfuser/${data_name}/${loss_function}_test
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
