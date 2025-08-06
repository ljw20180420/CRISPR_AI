#!/bin/bash

# for data_name in SX_spcas9 SX_spymac SX_ispymac
for data_name in SX_spcas9
do
    # Train
    # CRIformer
    ./run.py train --config AI/preprocess/train.yaml --train.output_dir /home/ljw/sdc1/CRISPR_results --train.trial_name default --dataset.name ${data_name} --model AI/preprocess/CRIformer/CRIformer.yaml
    # inDelphi
    ./run.py train --config AI/preprocess/train.yaml --train.output_dir /home/ljw/sdc1/CRISPR_results --train.trial_name default --dataset.name ${data_name} --model AI/preprocess/inDelphi/inDelphi.yaml
    # Lindel
    ./run.py train --config AI/preprocess/train.yaml --train.output_dir /home/ljw/sdc1/CRISPR_results --train.trial_name default --dataset.name ${data_name} --model AI/preprocess/Lindel/Lindel.yaml
    # DeepHF
    ./run.py train --config AI/preprocess/train.yaml --train.output_dir /home/ljw/sdc1/CRISPR_results --train.trial_name default --dataset.name ${data_name} --model AI/preprocess/DeepHF/DeepHF.yaml
    ./run.py train --config AI/preprocess/train.yaml --train.output_dir /home/ljw/sdc1/CRISPR_results --train.trial_name default --dataset.name ${data_name} --model AI/preprocess/DeepHF/CNN.yaml
    ./run.py train --config AI/preprocess/train.yaml --train.output_dir /home/ljw/sdc1/CRISPR_results --train.trial_name default --dataset.name ${data_name} --model AI/preprocess/DeepHF/MLP.yaml
    ./run.py train --config AI/preprocess/train.yaml --train.output_dir /home/ljw/sdc1/CRISPR_results --train.trial_name default --dataset.name ${data_name} --model AI/preprocess/DeepHF/XGBoost.yaml
    # Ridge is out of memory.
    # ./run.py train --config AI/preprocess/train.yaml --train.output_dir /home/ljw/sdc1/CRISPR_results --train.trial_name default --dataset.name ${data_name} --model AI/preprocess/DeepHF/Ridge.yaml
    # CRIfuser
    ./run.py train --config AI/preprocess/train.yaml --train.output_dir /home/ljw/sdc1/CRISPR_results --train.trial_name default --dataset.name ${data_name} --model AI/preprocess/CRIfuser/CRIfuser.yaml
    # FOREcasT
    ./run.py train --config AI/preprocess/train.yaml --train.output_dir /home/ljw/sdc1/CRISPR_results --train.trial_name default --dataset.name ${data_name} --model AI/preprocess/FOREcasT/FOREcasT.yaml

    # Test
    # CRIformer
    ./run.py test --config AI/preprocess/test.yaml --test.model_path /home/ljw/sdc1/CRISPR_results/CRIformer/CRIformer/${data_name}/default
    # inDelphi
    ./run.py test --config AI/preprocess/test.yaml --test.model_path /home/ljw/sdc1/CRISPR_results/inDelphi/inDelphi/${data_name}/default
    # Lindel
    ./run.py test --config AI/preprocess/test.yaml --test.model_path /home/ljw/sdc1/CRISPR_results/Lindel/Lindel/${data_name}/default
    # DeepHF
    ./run.py test --config AI/preprocess/test.yaml --test.model_path /home/ljw/sdc1/CRISPR_results/DeepHF/DeepHF/${data_name}/default
    ./run.py test --config AI/preprocess/test.yaml --test.model_path /home/ljw/sdc1/CRISPR_results/DeepHF/CNN/${data_name}/default
    ./run.py test --config AI/preprocess/test.yaml --test.model_path /home/ljw/sdc1/CRISPR_results/DeepHF/MLP/${data_name}/default
    ./run.py test --config AI/preprocess/test.yaml --test.model_path /home/ljw/sdc1/CRISPR_results/DeepHF/XGBoost/${data_name}/default
    # Ridge is out of memory.
    # ./run.py test --config AI/preprocess/test.yaml --test.model_path /home/ljw/sdc1/CRISPR_results/DeepHF/Ridge/${data_name}/default
    # CRIfuser
    ./run.py test --config AI/preprocess/test.yaml --test.model_path /home/ljw/sdc1/CRISPR_results/CRIfuser/CRIfuser/${data_name}/default
    # FOREcasT
    ./run.py test --config AI/preprocess/test.yaml --test.model_path /home/ljw/sdc1/CRISPR_results/FOREcasT/FOREcasT/${data_name}/default
done