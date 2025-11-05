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
hta_config=AI/hta.yaml
output_dir=${OUTPUT_DIR:-$HOME"/CRISPR_results"}/unit_test/default
test_config=AI/test.yaml
inference_config=AI/inference.yaml

for data_name in SX_spcas9
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
        checkpoints_path=${output_dir}/checkpoints/${preprocess}/${model_cls}/${data_name}/default
        logs_path=${output_dir}/logs/${preprocess}/${model_cls}/${data_name}/default

        title Train
        ./run.py train --config ${train_config} --train.output_dir ${output_dir} --train.trial_name default --train.num_epochs 1 --profiler.repeat 1 --dataset.data_file AI/dataset/test.json.gz --dataset.name ${data_name} --model ${model_config}

        title Eval
        ./run.py train --config ${train_config} --train.output_dir ${output_dir} --train.trial_name default --train.num_epochs 1 --train.evaluation_only true --dataset.data_file AI/dataset/test.json.gz --dataset.name ${data_name} --model ${model_config}

        title Hta
        ./run.py hta --config ${hta_config} --trace_dir ${logs_path}/profile

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

        title inference
        ./run.py inference --config ${inference_config} --output ${logs_path}/inference_output.csv --test.checkpoints_path ${checkpoints_path} --test.logs_path ${logs_path}
    done
done
