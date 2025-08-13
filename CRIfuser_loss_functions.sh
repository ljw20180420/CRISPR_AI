#!/bin/bash

# change to the dir of the script
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

train_config=AI/preprocess/train.yaml
output_dir=/home/ljw/sdc1/CRISPR_results
test_config=AI/preprocess/test.yaml
loss_weight=1.0

for data_name in SX_spcas9 SX_spymac SX_ispymac
do
    for loss_function in double_sample_negative_ELBO importance_sample_negative_ELBO forward_negative_ELBO reverse_negative_ELBO sample_CE non_sample_CE
    do
        ./run.py train --config ${train_config} --train.output_dir ${output_dir} --train.trial_name ${loss_function} --dataset.name ${data_name} --model AI/preprocess/CRIfuser/CRIfuser.yaml --model.loss_weights "{'${loss_function}': ${loss_weight}}"
    done
done