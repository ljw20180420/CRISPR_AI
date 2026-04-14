#!/bin/bash

# change to the dir of the script
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# change to the dir to the project
cd ../..

function title() {
    sharps="#################################"
    printf "\n%s\n%s\n%s\n" ${sharps} $1 ${sharps}
}

explain_config=AI/explain.yaml
output_dir=${OUTPUT_DIR:-$HOME"/CRISPR_results"}/formal/default
load_only=${load_only:-"true"}

for data_name in SX_spcas9 SX_spymac SX_ispymac
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
        checkpoints_path=${output_dir}/checkpoints/${preprocess}/${model_cls}/${data_name}/default
        logs_path=${output_dir}/logs/${preprocess}/${model_cls}/${data_name}/default

        title explain
        for shap_target in \
            small_indel \
            unilateral \
            large_indel \
            mmej
        do
            case ${model_cls} in
                CRIfuser)
                    ./run.py explain --config ${explain_config} --shap.load_only ${load_only} --shap.shap_target ${shap_target} --test.checkpoints_path ${checkpoints_path} --test.logs_path ${logs_path} --test.overwrite.model.init_args.eval_output_step 4 --test.overwrite.model.init_args.eval_output_batch_size 16 --dataset.name ${data_name}
                ;;
                *)
                    ./run.py explain --config ${explain_config} --shap.load_only ${load_only} --shap.shap_target ${shap_target} --test.checkpoints_path ${checkpoints_path} --test.logs_path ${logs_path} --dataset.name ${data_name}
                ;;
            esac
        done
    done
done
