#!/bin/bash

# change to the dir of the script
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

output_dir=/home/ljw/sdc1/CRIfuser_loss_functions
loss_weight=0.0001

for data_name in SX_spcas9 SX_spymac SX_ispymac
do
    for loss_function in double_sample_negative_ELBO importance_sample_negative_ELBO forward_negative_ELBO reverse_negative_ELBO sample_CE non_sample_CE
    do
        mkdir -p ${output_dir}/${loss_function}
        ./run.py --train --output_dir ${output_dir}/${loss_function} --dataset.owner ljw20180420 --dataset.data_name ${data_name} CRIfuser CRIfuser --loss_weights "{'${loss_function}': ${loss_weight}}"
    done
done