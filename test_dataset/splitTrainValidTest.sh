#!/bin/bash
# ljw2017@sjtu.edu.cn
# This file split the alignments file into train, valid, and test.
# Usage:
# splitTrainValidTest.sh <all.alg >train.alg 3>valid.alg 4>test.alg
# To change the percentage of valid data and test data:
# validPercent=0.15 testPercent=0.2 splitTrainValidTest.sh <all.alg >train.alg 3>valid.alg 4>test.alg

backTo3Lines()
{
    awk -F "\t" '{
        for (i = 1; i < NF - 2; ++i)
            printf("%s\t", $i)
        for (i = NF - 2; i <= NF; ++i)
            printf("%s\n", $i)
    }'
}

scriptPath=$(dirname $(realpath $0))

validPercent=${validPercent:-0.1}
testPercent=${testPercent:-0.1}
allTmp=$(mktemp)
sed -n 'N;N;s/\n/\t/g; p' >$allTmp
allCount=$(wc -l <$allTmp)
validCount=$(awk -v allCount=$allCount -v validPercent=$validPercent 'BEGIN{print int(allCount * validPercent)}')
testCount=$(awk -v allCount=$allCount -v testPercent=$testPercent 'BEGIN{print int(allCount * testPercent)}')
validTmp=$(mktemp)
shuf -n $validCount $allTmp >$validTmp
allTmp2=$(mktemp)
sort $allTmp | comm -23 - <(sort $validTmp) >$allTmp2
testTmp=$(mktemp)
shuf -n $testCount $allTmp2 >$testTmp
sort $allTmp2 | comm -23 - <(sort $testTmp) >$allTmp
backTo3Lines <$allTmp >&1
backTo3Lines <$validTmp >&3
backTo3Lines <$testTmp >&4
