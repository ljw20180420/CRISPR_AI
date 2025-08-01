#!/bin/bash

for data_name in SX_spcas9 SX_spymac SX_ispymac
do
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
done