#!/bin/bash

# change to the dir of the script
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir -p data_for_figs/benchmark
cp benchmark/default.csv data_for_figs/benchmark/default.csv

output_dir=${OUTPUT_DIR:-$HOME"/CRISPR_results"}
run_type="formal"
run_name="default"
trial_name="default"
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
        IFS=":" read preprocess model_cls <<<${pre_model}
        logs_path=${output_dir}/${run_type}/${run_name}/logs/${preprocess}/${model_cls}/${data_name}/${trial_name}

        for shap_target in \
            small_indel \
            unilateral \
            large_indel \
            mmej
        do
            mkdir -p data_for_figs/${model_cls}/${data_name}/${shap_target}
            cp ${logs_path}/explain/${shap_target}/explanation.h5 data_for_figs/${model_cls}/${data_name}/${shap_target}/explanation.h5
        done
    done
done

zip -r data_for_figs.zip data_for_figs 
