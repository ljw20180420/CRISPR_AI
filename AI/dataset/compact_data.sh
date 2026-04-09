#!/bin/bash

# change to the dir of the script
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

concat_algs()
{
    for author in SX SJLJH LE
    do
        for alg in $(find ${DATA_DIR}/$author/algs -type f)
        do
            bname=$(basename $alg)
            # ref1, ref2, cut1, cut2, author, file, ref1_end, ref2_start, random_insert, count, score
            zcat $alg | awk -F "\t" -v OFS="\t" -v author=$author -v file=${bname%.alg.gz} '
                NR % 3 == 1 {
                    cut1 = $16
                    cut2 = $17
                    ref1_end = $8
                    ref2_start = $11
                    random_insert = toupper($10)
                    count = $2
                    score = $3
                }
                NR % 3 == 2 {
                    gsub("-", "", $0)
                    ref1_len = match($0, /[acgtn]-*[acgtn]/)
                    ref1 = toupper(substr($0, 1, ref1_len))
                    ref2 = toupper(substr($0, ref1_len + 1))
                    print ref1, ref2, cut1, cut2 - ref1_len, author, file, ref1_end, ref2_start - ref1_len, random_insert, count, score
                }
            '
        done
    done
}

random_DNA()
{
    local length=$1
    local chars="ACGT"
    str=""
    for ((i = 0; i < ${length}; ++i)); do
        str+=${chars:RANDOM%${#chars}:1}
    done
    printf $str
}

random_scaffold()
{
    local scaffolds=(spcas9 spymac)
    printf ${scaffolds[RANDOM%2]}
}

concat_algs

# # Collate all data
# concat_algs |
# sort --parallel 24 -t $'\t' -k1,9 |
# ./to_json.awk -v min_score=0 |
# gzip > dataset.naapam.json.gz

# # Get small test dataset
# zcat dataset.naapam.json.gz |
# head -n 100 |
# gzip > test.naapam.json.gz

# # Generate inference data
# printf "ref,cut,scaffold\n" > inference.csv
# for i in {1..100}
# do
#     ref=$(random_DNA 104)"GG"$(random_DNA 94)
#     scaffold=$(random_scaffold)
#     printf "%s,100,%s\n" $ref $scaffold \
#         >> inference.csv
# done
# gzip --force inference.csv
