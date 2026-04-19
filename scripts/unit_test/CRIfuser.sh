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
output_dir=${OUTPUT_DIR:-$HOME"/CRISPR_results"}
run_type="unit_test"
run_name="CRIfuser"
test_config=AI/test.yaml
loss_weight=1.0

for data_name in SX_spcas9
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

        checkpoints_path=${output_dir}/${run_type}/${run_name}/checkpoints/CRIfuser/CRIfuser/${data_name}/${loss_function}
        logs_path=${output_dir}/${run_type}/${run_name}/logs/CRIfuser/CRIfuser/${data_name}/${loss_function}

        title Train
        ./run.py train \
            --config ${train_config} \
            --train.output_dir ${output_dir}/${run_type}/${run_name} \
            --train.trial_name ${loss_function} \
            --train.num_epochs 1 \
            --dataset.data_file AI/dataset/test.json.gz \
            --dataset.name ${data_name} \
            --model AI/preprocess/CRIfuser/CRIfuser.yaml \
            --model.loss_weights "{'${loss_function}': ${loss_weight}}"

        title Eval
        ./run.py train \
            --config ${train_config} \
            --train.output_dir ${output_dir}/${run_type}/${run_name} \
            --train.trial_name ${loss_function} \
            --train.num_epochs 1 \
            --train.evaluation_only true \
            --dataset.data_file AI/dataset/test.json.gz \
            --dataset.name ${data_name} \
            --model AI/preprocess/CRIfuser/CRIfuser.yaml \
            --model.loss_weights "{'${loss_function}': ${loss_weight}}"

        title Test
        for target in \
            CrossEntropy \
            NonZeroCrossEntropy \
            NonWildTypeCrossEntropy \
            NonZeroNonWildTypeCrossEntropy \
            GreatestCommonCrossEntropy \
            Likelihood \
            Pearson \
            MSE \
            SymKL
        do
            if [[ "${target}" == "Likelihood" ]] || [[ "${target}" == "Pearson" ]]
            then
                maximize_target=true
            else
                maximize_target=false
            fi
            ./run.py test \
                --config ${test_config} \
                --checkpoints_path ${checkpoints_path} \
                --logs_path ${logs_path} \
                --target ${target} \
                --maximize_target ${maximize_target}
        done
    done
done
