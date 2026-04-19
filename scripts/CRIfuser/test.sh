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
output_dir=${OUTPUT_DIR:-$HOME"/CRISPR_results"}
run_type="formal"
run_name="CRIfuser"

for data_name in SX_spcas9 SX_spymac SX_ispymac
do
    title ${data_name}
    for loss_function in \
        double_sample_negative_ELBO \
        importance_sample_negative_ELBO \
        forward_negative_ELBO \
        reverse_negative_ELBO \
        sample_CE \
        non_sample_CE
    do
        title ${loss_function}

        checkpoints_path=${output_dir}/${run_type}/${run_name}/checkpoints/CRIfuser/CRIfuser/${data_name}/${loss_function}
        logs_path=${output_dir}/${run_type}/${run_name}/logs/CRIfuser/CRIfuser/${data_name}/${loss_function}

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