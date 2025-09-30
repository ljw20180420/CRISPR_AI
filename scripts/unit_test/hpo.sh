#!/bin/bash

# change to the dir of the script
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# change to the dir to the project
cd ../..

hpo_config="AI/hpo.yaml"
output_dir=${OUTPUT_DIR:-${HOME}"/CRISPR_results"}/unit_test/hpo
trial_name="trial"

for data_name in SX_spcas9
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
        IFS=":" read preprocess model_cls <<<${pre_model}
        # trial_name will be appended by trial id like trial_name-0, trial_name-1 and so on.
        ./run.py hpo \
            --config ${hpo_config} \
            --hpo.target GreatestCommonCrossEntropy \
            --hpo.study_name study \
            --hpo.n_trials 2 \
            --train.train.output_dir ${output_dir} \
            --train.train.trial_name ${trial_name} \
            --train.train.num_epochs 2 \
            --train.dataset,data_file AI/dataset/test.json.gz \
            --train.dataset.name ${data_name} \
    done
done