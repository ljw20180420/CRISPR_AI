#!/bin/bash

# change to the dir of the script
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# change to the dir to the project
cd ../..

output_dir=${OUTPUT_DIR:-${HOME}"/CRISPR_results"}/unit_test/hpo

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
        # Hpo
        ./hpo.py --output_dir ${output_dir} --preprocess ${preprocess} --model_type ${model_type} --data_file AI/dataset/test.json.gz --data_name ${data_name} --batch_size 100 --num_epochs 2 --target_metric GreatestCommonCrossEntropy --sampler TPESampler --pruner SuccessiveHalvingPruner --study_name hpo --n_trials 2 --load_if_exists false
    done
done