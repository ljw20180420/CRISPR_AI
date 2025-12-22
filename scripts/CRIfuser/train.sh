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
output_dir=${OUTPUT_DIR:-$HOME"/CRISPR_results"}/formal/CRIfuser
loss_weight=1.0

for data_name in SX_spcas9 SX_spymac SX_ispymac
do
    title ${data_name}
    for loss_function in \
        double_sample_negative_ELBO \
        importance_sample_negative_ELBO \
        forward_negative_ELBO \
        reverse_negative_ELBO \
        sample_CE non_sample_CE
    do
        title ${loss_function}

        title Train
        ./run.py train --config ${train_config} --train.output_dir ${output_dir} --train.trial_name ${loss_function} --train.evaluation_only false --dataset.name ${data_name} --model AI/preprocess/CRIfuser/CRIfuser.yaml --model.loss_weights "{'${loss_function}': ${loss_weight}}"
    done
done