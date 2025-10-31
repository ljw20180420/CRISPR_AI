#!/bin/bash

preprocess=CRIfuser
model_cls=CRIfuser
data_name=SX_ispymac # SX_spcas9, SX_spymac, SX_ispymac

for path in $(find /home/ljw/sdc1/CRISPR_results/formal/default/checkpoints/${preprocess}/${model_cls}/${data_name}/default -name GreatestCommonCrossEntropy.yaml)
do
    cp AI/metric/GreatestCommonCrossEntropy.yaml $path
done
