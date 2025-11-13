# 、AIdit_Cas9运行代码

# 根据 gRNA 序列在基因组中匹配局部序列

# -*-coding: utf-8 -*-

import os
import re
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


# 检查文件是否存在，存在删除
def is_Exist_file(path):
    import os

    if os.path.exists(path):
        os.remove(path)


def mkdir(path):
    import os

    path = path.strip()  # 去除首位空格
    path = path.rstrip("\\")  # 去除尾部 \ 符号
    isExists = os.path.exists(path)  # 判断路径是否存在
    # 判断结果
    if not isExists:
        os.makedirs(path)  # 如果不存在则创建目录
        print(path + " 创建成功")
    else:
        print(path + " 目录已存在")  # 如果目录存在则不创建，并提示目录已存在


## Match sequencing from hg19
def matching_sgRNA_from_hg19(hg19_path, gRNA_list, antisense=False, size=20):
    import time
    import gzip
    from Bio import SeqIO
    from Bio.Seq import reverse_complement

    # hg19_path = './0_Human Genome/GRCh37_latest_genomic.fna.gz'
    start = time.time()
    with gzip.open(hg19_path, "rt") as handle:
        i = 0
        data_dict = {}
        for seq_record in SeqIO.parse(handle, "fasta"):
            string = str(seq_record.seq).upper()
            i += 1
            for gRNA in gRNA_list:
                if not antisense:
                    pattern = gRNA.upper() + ".GG"
                    # pattern = gRNA.upper()
                else:
                    pattern = "CC." + reverse_complement(gRNA).upper()  ## 反向互补
                    # pattern = reverse_complement(gRNA).upper()  ## 反向互补

                des0 = seq_record.description.split(",")[0]
                des1 = " ".join(des0.split(" ")[1:5])
                key = "%s %s-" % (i, des1) + gRNA
                ## obtain Up, Down Sequence
                value = re.findall(pattern, string)
                j = 0
                new_value = []
                for a in value:
                    i_index = string.index(a, j)
                    new_value.append(string[(i_index - size) : (i_index + 23 + size)])
                    j = i_index + 1
                data_dict[key] = new_value
            # print('Description:', seq_record.description)

    end = time.time()
    print("Matching is Over. Using Time: %s" % (round(end - start, 3)))
    return data_dict


## 解析 data_dict
def parse_gRNA_data_dict(data_dict):
    sum_dict = {}
    for key, value in data_dict.items():
        ps = key.split("-")
        chr_n = " ".join(ps[0].split(" ")[1:])
        p = key.split("-")[1]
        if p not in sum_dict:
            sum_dict[p] = [(chr_n, value)]
        else:
            if len(value) != 0:
                element0 = sum_dict[p][-1]
                if len(element0[1]) == 0:
                    sum_dict[p].remove(element0)
                else:
                    pass

                element = (chr_n, value)
                sum_dict[p].append(element)
            else:
                pass

    ## generate DataFrame

    gRNA_list = []

    soure_list = []

    ref_gRNA_list = []

    for gRNA, value in sum_dict.items():

        for element in value:

            gRNA_list.append(gRNA)

            soure_list.append(element[0])

            if len(element[1]) == 0:

                ref_gRNA_list.append("")

            else:

                ref_gRNA_list.append(element[1][0])

    data = pd.DataFrame(
        {"gRNASeq": gRNA_list, "chromosome": soure_list, "ref_gRNASeq": ref_gRNA_list}
    )

    ## drop_duplicates

    data.drop_duplicates(inplace=True)

    data.reset_index(drop=True, inplace=True)

    return data


## 主函数：分别匹配&解析正义链和反义链


def main_match(hg19_path, gRNA_list, size=20):

    from Bio.Seq import reverse_complement

    ######### 正义链匹配 #########

    ## hg19 matching

    data_dict = matching_sgRNA_from_hg19(hg19_path, gRNA_list, False, size)

    ## 解析 data_dict

    data = parse_gRNA_data_dict(data_dict)

    ##############################

    ## check match result

    ## split sgRNA: Match & NonMatch

    mdata = data.loc[data["ref_gRNASeq"] != "", :]  # Match data

    nonmdata = data.loc[data["ref_gRNASeq"] == "", :]  # NonMatch data

    mdata.reset_index(drop=True, inplace=True)

    nonmdata.reset_index(drop=True, inplace=True)

    mdata["ref_Strand"] = "+"

    print("Total sgRNA number: %s" % (len(data)))

    print("Sense Matched sgRNA number: %s" % (len(mdata)))

    ######### 反义链匹配 #########

    # hg19 matching

    data_dict = matching_sgRNA_from_hg19(hg19_path, gRNA_list, True, size)

    ## 解析 data_dict

    patch_data = parse_gRNA_data_dict(data_dict)

    patch_data = patch_data.loc[patch_data["ref_gRNASeq"] != "", :]  # Match data

    if len(patch_data) != 0:

        patch_data["ref_gRNASeq"] = patch_data["ref_gRNASeq"].apply(
            lambda x: reverse_complement(x)
        )

        patch_data["ref_Strand"] = "-"

    else:

        pass

    print("Not SensecMatched sgRNA number: %s" % (len(patch_data)))

    ## Concat

    data = pd.concat([mdata, patch_data], axis=0)

    data.reset_index(drop=True, inplace=True)

    data["ref_gRNAUp"] = data["ref_gRNASeq"].apply(lambda x: x[:size])

    data["ref_gRNATarget"] = data["ref_gRNASeq"].apply(lambda x: x[size : (size + 20)])

    data["ref_PAM"] = data["ref_gRNASeq"].apply(lambda x: x[(size + 20) : (size + 23)])

    data["ref_gRNADown"] = data["ref_gRNASeq"].apply(lambda x: x[(size + 23) :])

    return data


# 1、SpCas9 on-target 活性预测

# 1.1、on-target 特征工程

import os

import numpy as np

import warnings

warnings.filterwarnings("ignore")


def is_Exist_file(path):

    import os

    import shutil

    if os.path.exists(path):

        try:

            shutil.rmtree(path)

        except NotADirectoryError as e:

            os.remove(path)


def mkdir(path):

    path = path.strip()

    path = path.rstrip("\\")

    isExists = os.path.exists(path)

    if not isExists:

        os.makedirs(path)

        print(path + " Create directory successfully.")

    else:

        print(path + " the directory exists.")


# input: target directory

# output: complete path (path + file name)


def walk(path):

    input_path_list = []

    if not os.path.exists(path):

        return -1

    for root, dirs, names in os.walk(path):

        for filename in names:

            input_path = os.path.join(root, filename)

            input_path_list.append(input_path)

    return input_path_list


# input： sequence features


def find_all(sub, s):

    index = s.find(sub)

    feat_one = np.zeros(len(s))

    while index != -1:

        feat_one[index] = 1

        index = s.find(sub, index + 1)

    return feat_one


# obtain single sequence


def obtain_each_seq_data(seq):

    A_array = find_all("A", seq)

    G_array = find_all("G", seq)

    C_array = find_all("C", seq)

    T_array = find_all("T", seq)

    one_sample = np.array([A_array, G_array, C_array, T_array])

    # print(one_sample.shape)

    return one_sample


# obtain sequence data for dataframe


def obtain_Sequence_data(data, layer_label="1D"):
    """

    input: dataframe with 'target sequence' column

    (63bp: 20bp downstream + 20bp target + 3bp pam + 20bp upstream)

    """

    x_data = []

    for i, row in data.iterrows():

        try:

            seq = row["target sequence"]

            assert seq[41:43] == "GG"

            one_sample = obtain_each_seq_data(seq)

        except AttributeError as e:

            raise e

        if layer_label == "1D":  # for LSTM or Conv1D, shape=(sample, step, feature)

            one_sample_T = one_sample.T

            x_data.append(one_sample_T)

        else:

            x_data.append(one_sample)

    x_data = np.array(x_data)

    if layer_label == "2D":  # for Conv2D shape=(sample, rows, cols, channels)

        x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], x_data.shape[2], 1)

    else:

        pass  # for LSTM or Conv1D: shape=(sample, step, feature)

    x_data = x_data.astype("float32")

    # print('After transformation, x_data.shape:', x_data.shape)

    return x_data


# for single sequence


def obtain_single_sequence_data(seq):

    x_data = []

    assert (seq[41:43] == "GG") & (len(seq) == 63)

    one_sample = obtain_each_seq_data(seq)

    one_sample_T = one_sample.T

    x_data.append(one_sample_T)

    x_data = np.array(x_data)

    x_data = x_data.astype("float32")

    return x_data


# 1.2、SpCas9 on-target 活性预测

import sys

from on_target_features import *

import warnings

warnings.filterwarnings("ignore")


# for batch sequence

# data with 'target sequence' column


def predict_batch_data_2(
    model_directory, rnn_params, model_func, input_path, output_path, seq_len
):

    # load model

    model = model_func(rnn_params, seq_len)

    model.load_weights(model_directory).expect_partial()

    is_Exist_file(output_path)

    with open(output_path, "a") as a:

        with open(input_path, "r") as f:

            a.write("target sequence\tefficiency\n")

            next(f)

            batch_n = 100000

            i = 0

            x_data = []

            seq_list = []

            for line in f:

                i += 1

                line = line.strip(" ").strip("\n")

                seq = line.split("\t")[0]

                if i <= batch_n:

                    # assert (seq[41:43] == 'GG') & (len(seq) == 63)

                    one_sample = obtain_each_seq_data(seq)

                    one_sample_T = one_sample.T

                    seq_list.append(seq)

                    x_data.append(one_sample_T)

                else:

                    # predict

                    x_data = np.array(x_data)

                    x_data = x_data.astype("float32")

                    ypred = model.predict(x_data)

                    for index, seq in enumerate(seq_list):

                        eff = ypred[index][0]

                        a.write("%s\t%s\n" % (seq, eff))

                    # Re-initial

                    i, x_data, seq_list = 0, [], []

                    # assert (seq[41:43] == 'GG') & (len(seq) == 63)

                    one_sample = obtain_each_seq_data(seq)

                    one_sample_T = one_sample.T

                    seq_list.append(seq)

                    x_data.append(one_sample_T)

            # predict

            x_data = np.array(x_data)

            x_data = x_data.astype("float32")

            ypred = model.predict(x_data)

            for index, seq in enumerate(seq_list):

                eff = ypred[index][0]

                a.write("%s\t%s\n" % (seq, eff))


# predict


def main_on_predict(
    cell_line,
    model_directory,
    rnn_params,
    model_func,
    input_path,
    output_directory,
    seq_len,
):

    # model_directory = model_directory_dict[cell_line]

    mkdir(output_directory)

    output_path = output_directory + "/predicted_result_Aidit_ON_%s.txt" % cell_line

    predict_batch_data_2(
        model_directory, rnn_params, model_func, input_path, output_path, seq_len
    )

    import pandas as pd

    csv_output_path = output_directory + "/predicted_result_Aidit_ON_%s.csv" % cell_line

    data = pd.read_csv(output_path, sep="\t")

    data.to_csv(csv_output_path, index=False)

    is_Exist_file(output_path)


# SpCas9 off-target 活性预测

# 2.1、SpCas9 off-target 特征工程

# 'target sequence' column: 63bp wild-type sequence (20bp downstream + 20bp target + 3bp PAM + 20bp upstream);

# 'off-target sequence' column: 63bp off-target sequence (20bp downstream + 20bp off-target + 3bp PAM + 20bp upstream).

import os

import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings("ignore")


def is_Exist_file(path):

    import os

    import shutil

    if os.path.exists(path):

        try:

            shutil.rmtree(path)

        except NotADirectoryError as e:

            os.remove(path)


def mkdir(path):

    import os

    path = path.strip()

    path = path.rstrip("\\")

    isExists = os.path.exists(path)

    if not isExists:

        os.makedirs(path)

    else:

        print(path + " 目录已存在")


# 计算 a pair of sequences 之间的 mismatch 的个数


def compute_mismatch_number(seq1, seq2):

    mismatch_num = 0

    for index, nucle1 in enumerate(seq1):

        nucle2 = seq2[index]

        if nucle1 != nucle2:

            mismatch_num += 1

        else:

            pass

    return mismatch_num


# gRNASeq & offSeq 比对


def alignment_on_off_Sequence(gRNASeq, offSeq):

    gRNASeq = gRNASeq.upper()

    offSeq = offSeq.upper()

    align = []

    for index, nucle0 in enumerate(gRNASeq):

        nucle1 = offSeq[index]

        if nucle1 != "-":

            align.append(nucle0 + nucle1)

        else:

            align.append(nucle0 + nucle0)

    return "-".join(align)


# alignment: on-off deletion sequence


def alignment_on_off_deletion_sequence(new_offSeq_Target):

    on_off_deltSeq = ""

    for nucle in new_offSeq_Target:

        if nucle != "-":

            on_off_deltSeq = on_off_deltSeq + "."

        else:

            on_off_deltSeq = on_off_deltSeq + "-"

    return on_off_deltSeq


# 比对确定 off-target insertion sequence


def alignment_on_off_insertion_sequence(offSeq_Target):

    import re

    inser_nucles = re.findall("[acgt]", offSeq_Target)

    inser_nucles = list(set(inser_nucles))

    ##

    inserSeq = ""

    for index, nucle in enumerate(offSeq_Target):

        if nucle not in inser_nucles:

            inserSeq = inserSeq + "."

        else:

            inserSeq = inserSeq + nucle.upper()

    return inserSeq


# off-target mismatch/insertion/deletion modeling data


def main_off_target_Modeling_data(data, mut_type="mismatch"):

    import copy

    off_data = copy.deepcopy(data)

    off_data["gRNASeq"] = off_data["target sequence"].apply(lambda x: x[20:43])

    off_data["offSeq_23bp"] = off_data["off-target sequence"].apply(lambda x: x[20:43])

    off_data["PAM-NN"] = off_data["off-target sequence"].apply(lambda x: x[41:43])

    off_data["on_off_alignSeq"] = off_data.apply(
        lambda row: alignment_on_off_Sequence(row["gRNASeq"], row["offSeq_23bp"]),
        axis=1,
    )

    if mut_type == "mismatch":

        # compute mismatch number

        off_data["up_mismatch_num"] = 0

        off_data["core_mismatch_num"] = off_data.apply(
            lambda row: compute_mismatch_number(
                row["target sequence"][20:43], row["off-target sequence"][20:43]
            ),
            axis=1,
        )

        off_data["down_mismatch_num"] = 0

        cols = [
            "target sequence",
            "off-target sequence",
            "gRNASeq",
            "offSeq_23bp",
            "PAM-NN",
            "on_off_alignSeq",
            "up_mismatch_num",
            "core_mismatch_num",
            "down_mismatch_num",
            "on_pred",
            "off_pred",
        ]

    elif mut_type == "deletion":

        off_data["on_off_deltSeq"] = off_data["off-target sequence"].apply(
            lambda x: alignment_on_off_deletion_sequence(x[20:43])
        )

        cols = [
            "target sequence",
            "off-target sequence",
            "gRNASeq",
            "offSeq_23bp",
            "PAM-NN",
            "on_off_alignSeq",
            "on_off_deltSeq",
            "on_pred",
            "off_pred",
        ]

    elif mut_type == "insertion":

        off_data["on_off_inserSeq"] = off_data["off-target sequence"].apply(
            lambda x: alignment_on_off_insertion_sequence(x[20:43])
        )

        cols = [
            "target sequence",
            "off-target sequence",
            "gRNASeq",
            "offSeq_23bp",
            "PAM-NN",
            "on_off_alignSeq",
            "on_off_inserSeq",
            "on_pred",
            "off_pred",
        ]

    else:

        print(
            "Mutation type not in ['mismatch', 'insertion', 'deletion']. Please check and try again."
        )

        cols = [
            "target sequence",
            "off-target sequence",
            "gRNASeq",
            "offSeq_23bp",
            "PAM-NN",
            "on_off_alignSeq",
            "on_pred",
            "off_pred",
        ]

    data = off_data[cols]

    return data


# ********************* Feature one-hot Encoding ***********************

# 1、序列特征输入： 序列特征

# 生成 Seequence 数据


def find_all(sub, s):

    index = s.find(sub)

    feat_one = np.zeros(len(s))

    while index != -1:

        feat_one[index] = 1

        index = s.find(sub, index + 1)

    return feat_one


# 获取单样本序列数据


def obtain_each_seq_data(seq):

    A_array = find_all("A", seq)

    G_array = find_all("G", seq)

    C_array = find_all("C", seq)

    T_array = find_all("T", seq)

    one_sample = np.array([A_array, G_array, C_array, T_array])

    # print(one_sample.shape)

    return one_sample


#  获取序列数据

# 参数说明：

# data：输入的数据，要求含有 gRNA_28bp or gRNASeq_63bp 列名，该列为原始 DNA 序列

# 输出：特征数据 {'data': data}


def obtain_sequence_flatten_data(data, seq_len=23, col="offSeq_23bp"):

    x_data = []

    for i, row in data.iterrows():

        seq = row[col]

        one_sample = obtain_each_seq_data(seq)

        one_sample_reshape = one_sample.T.reshape(seq_len * 4)

        # print(one_sample_reshape.shape)

        x_data.append(one_sample_reshape)

    # reshape

    x_data = np.array(x_data)

    x_data = x_data.astype("float32")

    return x_data


# 2、获得 PAM-NN 特征


def obtain_PAM_Feature(
    pam_nn, pam_feats=["GG", "AG", "GT", "GC", "GA", "TG", "CG", "other"]
):
    """

    pam_feats = ['GG', 'AG', 'GT', 'GC', 'GA', 'TG', 'CG', 'other']

    pam_nn = 'GG'

    pam_list = obtain_PAM_Feature(pam_nn, pam_feats)

    """

    pam_dict = {}

    for pam in pam_feats:

        pam_dict[pam] = 0

    if pam_nn in pam_dict:

        pam_dict[pam_nn] = 1

    else:

        pam_dict["other"] = 1

    # print(pam_dict)

    pam_list = []

    for pam in pam_feats:

        pam_list.append(pam_dict[pam])

    return pam_list


# 获得 PAM-NN 特征


def main_pam_data(data, pam_feats=["GG", "AG", "GT", "GC", "GA", "TG", "CG", "other"]):

    pam_data = []

    for index, row in data.iterrows():

        pam_nn = row["PAM-NN"]

        pam_list = obtain_PAM_Feature(pam_nn, pam_feats)

        pam_data.append(pam_list)

    ## pam data

    pam_data = np.array(pam_data)

    pam_data = pam_data.astype("float32")

    return pam_data


# 3、分解到每一个位置的 on-off mismatch feature

# 1、on-off alignment for position-substitution


def helper_each_position_alignSeq(one_pos_alignSeq):

    align_order = [
        "AC",
        "AG",
        "AT",
        "CA",
        "CG",
        "CT",
        "GA",
        "GC",
        "GT",
        "TA",
        "TC",
        "TG",
    ]

    align_list = []

    for one_align in align_order:

        if one_align == one_pos_alignSeq:

            align_list.append(1)

        else:

            align_list.append(0)

    return align_list


# one mismatch alignment sequence


def helper_one_alignSeq(alignSeq):
    """

    alignSeq = 'TT-TG-GC-CA-AG-GC-CA-AC-CC-AA-AA-CC-CC-CC-CC-AA-TT-GG-AA-AA-GG-GG-GG'

    all_align1 = helper_one_alignSeq(alignSeq)

    print(all_align1)

    """

    alignSeq_list = alignSeq.split("-")

    all_align_list = []

    for alignSeq in alignSeq_list:

        align_list = helper_each_position_alignSeq(alignSeq)

        all_align_list.append(align_list)

    all_align = np.array(all_align_list).T

    return all_align


# 获得 mismatch alignment feature


def main_mismatch_alignment_features(data):

    align_data = []

    for index, row in data.iterrows():

        alignSeq = row["on_off_alignSeq"]

        all_align = helper_one_alignSeq(alignSeq)

        all_align = all_align.T.reshape(all_align.shape[0] * all_align.shape[1])

        align_data.append(all_align)

    # align data

    align_data = np.array(align_data)

    align_data = align_data.astype("float32")

    return align_data


# 2、on-off alignment for only position


def helper_one_alignSeq_with_only_position(alignSeq):
    """

    alignSeq = 'TT-TG-GC-CA-AG-GC-CA-AC-CC-AA-AA-CC-CC-CC-CC-AA-TT-GG-AA-AA-GG-GG-GG'

    all_align_list2 = helper_one_alignSeq_with_only_position(alignSeq)

    print(all_align_list2)

    """

    alignSeq_list = alignSeq.split("-")

    all_align_list = []

    for alignSeq in alignSeq_list:

        if alignSeq[0] == alignSeq[1]:

            all_align_list.append(0)

        else:

            all_align_list.append(1)

    return all_align_list


# 获得 mismatch alignment feature with only position


def main_mismatch_alignment_features_with_only_position(data):

    align_data = []

    for index, row in data.iterrows():

        alignSeq = row["on_off_alignSeq"]

        all_align_list = helper_one_alignSeq_with_only_position(alignSeq)

        align_data.append(all_align_list)

    # align data

    align_data = np.array(align_data)

    align_data = align_data.astype("float32")

    return align_data


# 4、获得 on-off deletion position distribution


def helper_on_off_deletion_position(on_off_deltSeq):

    deltSeq_list = []

    for m in on_off_deltSeq:

        if m == ".":

            deltSeq_list.append(0)

        else:

            deltSeq_list.append(1)

    return deltSeq_list


# 获得 on-off deletion position feature


def main_on_off_deletion_position(data):

    delt_data = []

    for index, row in data.iterrows():

        on_off_deltSeq = row["on_off_deltSeq"]

        deltSeq_list = helper_on_off_deletion_position(on_off_deltSeq)

        delt_data.append(deltSeq_list)

    ## delt data

    delt_data = np.array(delt_data)

    delt_data = delt_data.astype("float32")

    return delt_data


# 5、获得 on-off insertion position-nucleotide type


def help_on_off_insertion(on_off_inserSeq):

    ref_dict = {
        ".": [0, 0, 0, 0],
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
    }

    inserSeq_list = []

    for m in on_off_inserSeq[1:]:

        inserSeq_list.append(ref_dict[m])

    return inserSeq_list


# 获得 on-off insertion feature


def main_on_off_insertion_feature(data):

    inser_data = []

    for index, row in data.iterrows():

        on_off_inserSeq = row["on_off_inserSeq"]

        inserSeq_list = help_on_off_insertion(on_off_inserSeq)

        inserSeq = np.array(inserSeq_list)

        inserSeq = inserSeq.reshape(4 * (len(on_off_inserSeq) - 1))

        inser_data.append(inserSeq)

    # inser data

    inser_data = np.array(inser_data)

    inser_data = inser_data.astype("float32")

    return inser_data


# 仅考虑 insertion position


def help_on_off_insertion_position(on_off_inserSeq):

    inserSeq_pos = []

    for m in on_off_inserSeq[1:]:

        if m == ".":

            inserSeq_pos.append(0)

        else:

            inserSeq_pos.append(1)

    return inserSeq_pos


# 获得 on-off insertion feature position


def main_on_off_insertion_feature_woth_only_position(data):

    inser_data = []

    for index, row in data.iterrows():

        on_off_inserSeq = row["on_off_inserSeq"]

        inserSeq_pos = help_on_off_insertion_position(on_off_inserSeq)

        inser_data.append(inserSeq_pos)

    ## inser data

    inser_data = np.array(inser_data)

    inser_data = inser_data.astype("float32")

    return inser_data


# mismatch

# 得到 off-target mismatch Feature Engineering

# must have: offSeq_63bp/offSeq_28bp, gRNASeq

# selective: PAM-NN,  on_off_alignSeq

# nparray_concat_to_one


def array_concat_to_one(collect_feat_data_dict):

    data = pd.DataFrame()

    for feat_label, array in collect_feat_data_dict.items():

        df_array = pd.DataFrame(array)

        cols_n = df_array.shape[1]

        cols = [feat_label + "_%s" % (i + 1) for i in range(cols_n)]

        df_array.columns = cols

        data = pd.concat([data, df_array], axis=1)

    return data


# deletion -- UPDATE

# 得到 off-target deletion Feature Engineering

# must have: offSeq_63bp/offSeq_28bp, gRNASeq

# selective: PAM-NN,  on_off_alignSeq, on_off_deltSeq

# feat_label 表示

# '+P': 'PAM-NN';

# '+M': 'on_off_alignSeq';

# '+Mp': 'on_off_alignSeq' with only position;

# '+D': 'on_off_deltSeq';

# '+P+M': 'PAM-NN + on_off_alignSeq';

# '+P+Mp': 'PAM-NN + on_off_alignSeq' with only position;

# '+P+D': 'PAM-NN + on_off_deltSeq';

# '+M+D': 'on_off_alignSeq + on_off_deltSeq';

# '+Mp+D': 'on_off_alignSeq + on_off_deltSeq' with only mismatch position;

# '+P+M+D': 'PAM-NN + on_off_alignSeq + on_off_deltSeq';

# '+P+Mp+D': 'PAM-NN + on_off_alignSeq + on_off_deltSeq' with mismath position;

# '+N': None.


def off_target_mismatch_feature_engineering(data, fixed_feat, feat_label):

    # geting features: 'gRNASeq', 'PAM-NN', 'on_off_alignSeq'

    data = main_off_target_Modeling_data(data, mut_type="mismatch")

    pam_feats = ["GG", "AG", "GT", "GC", "GA", "TG", "CG", "other"]

    collect_feat_data_dict = {}

    # fixed feature list

    if fixed_feat == "seq_feat":

        x_data1 = obtain_sequence_flatten_data(data, seq_len=23, col="offSeq_23bp")

        x_data2 = obtain_sequence_flatten_data(data, seq_len=23, col="gRNASeq")

        collect_feat_data_dict["offSeq"] = x_data1

        collect_feat_data_dict["gRNASeq"] = x_data2

    elif fixed_feat == "mismatch_num":

        x_data3 = np.array(
            data[["up_mismatch_num", "core_mismatch_num", "down_mismatch_num"]]
        )

        collect_feat_data_dict["mismatch_num_region"] = x_data3

    elif fixed_feat == "pred_feat":

        pred_data = np.array(data[["on_pred", "off_pred"]])

        collect_feat_data_dict["pred_feat"] = pred_data

    elif fixed_feat == "seq_feat+mismatch_num":

        x_data1 = obtain_sequence_flatten_data(data, seq_len=23, col="offSeq_23bp")

        x_data2 = obtain_sequence_flatten_data(data, seq_len=23, col="gRNASeq")

        x_data3 = np.array(
            data[["up_mismatch_num", "core_mismatch_num", "down_mismatch_num"]]
        )

        collect_feat_data_dict["offSeq"] = x_data1

        collect_feat_data_dict["gRNASeq"] = x_data2

        collect_feat_data_dict["mismatch_num_region"] = x_data3

    elif fixed_feat == "pred_feat+mismatch_num":

        pred_data = np.array(data[["on_pred", "off_pred"]])

        x_data3 = np.array(
            data[["up_mismatch_num", "core_mismatch_num", "down_mismatch_num"]]
        )

        collect_feat_data_dict["pred_feat"] = pred_data

        collect_feat_data_dict["mismatch_num_region"] = x_data3

    elif fixed_feat == "all":

        x_data1 = obtain_sequence_flatten_data(data, seq_len=23, col="offSeq_23bp")

        x_data2 = obtain_sequence_flatten_data(data, seq_len=23, col="gRNASeq")

        x_data3 = np.array(
            data[["up_mismatch_num", "core_mismatch_num", "down_mismatch_num"]]
        )

        pred_data = np.array(data[["on_pred", "off_pred"]])

        collect_feat_data_dict["offSeq"] = x_data1

        collect_feat_data_dict["gRNASeq"] = x_data2

        collect_feat_data_dict["mismatch_num_region"] = x_data3

        collect_feat_data_dict["pred_feat"] = pred_data

    else:  # None

        pass

    # Additional features

    if feat_label == "+P":

        pam_data = main_pam_data(data, pam_feats)

        collect_feat_data_dict["+P"] = pam_data

    elif feat_label == "+M":

        align_data = main_mismatch_alignment_features(data)

        collect_feat_data_dict["+M"] = align_data

    elif feat_label == "+Mp":  # consider mismatch position

        align_data = main_mismatch_alignment_features_with_only_position(data)

        collect_feat_data_dict["+Mp"] = align_data

    elif feat_label == "+P+M":

        pam_data = main_pam_data(data, pam_feats)

        align_data = main_mismatch_alignment_features(data)

        collect_feat_data_dict["+P"] = pam_data

        collect_feat_data_dict["+M"] = align_data

    elif feat_label == "+P+Mp":

        pam_data = main_pam_data(data, pam_feats)

        align_data = main_mismatch_alignment_features_with_only_position(data)

        collect_feat_data_dict["+P"] = pam_data

        collect_feat_data_dict["+Mp"] = align_data

    else:  # None

        pass

    # feature concating

    xdata = array_concat_to_one(collect_feat_data_dict)

    xdata = np.array(xdata)

    return xdata


# 2.2、SpCas9 off-target 活性预测

from off_target_features import *

import sys

import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings("ignore")


# step 1: to predict target sequence on-target scores

# obtain sequence data for dataframe


def obtain_Sequence_data(data, layer_label="1D"):
    """

    input: dataframe with 'target sequence' column

    (63bp: 20bp downstream + 20bp target + 3bp pam + 20bp upstream)

    """

    x_data = []

    for i, row in data.iterrows():

        try:

            seq = row["target sequence"]

            # assert seq[41:43] == "GG"

            one_sample = obtain_each_seq_data(seq)

        except AttributeError as e:

            raise e

        if layer_label == "1D":  # for LSTM or Conv1D, shape=(sample, step, feature)

            one_sample_T = one_sample.T

            x_data.append(one_sample_T)

        else:

            x_data.append(one_sample)

    x_data = np.array(x_data)

    if layer_label == "2D":  # for Conv2D shape=(sample, rows, cols, channels)

        x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], x_data.shape[2], 1)

    else:

        pass  # for LSTM or Conv1D: shape=(sample, step, feature)

    x_data = x_data.astype("float32")

    # print('After transformation, x_data.shape:', x_data.shape)

    return x_data


# get on-target features


def get_on_target_features(model_directory, rnn_params, model_func, data):

    wtdata = data[["target sequence"]]

    otdata = data[["off-target sequence"]]

    otdata.rename(columns={"off-target sequence": "target sequence"}, inplace=True)

    wtdata.drop_duplicates(inplace=True)

    otdata.drop_duplicates(inplace=True)

    wtdata.reset_index(drop=True, inplace=True)

    otdata.reset_index(drop=True, inplace=True)

    # on-target score

    # model_directory = on_model_directory_dict[cell]

    model = model_func(rnn_params, seq_len=63)

    model.load_weights(model_directory).expect_partial()

    x_wt = obtain_Sequence_data(wtdata, layer_label="1D")

    x_ot = obtain_Sequence_data(otdata, layer_label="1D")

    ypred_wt = model.predict(x_wt)

    ypred_ot = model.predict(x_ot)

    wtdata["on_pred"] = ypred_wt

    otdata["off_pred"] = ypred_ot

    otdata.rename(columns={"target sequence": "off-target sequence"}, inplace=True)

    # merge

    data = pd.merge(data, wtdata, how="left", on="target sequence")

    data = pd.merge(data, otdata, how="left", on="off-target sequence")

    data.reset_index(drop=True, inplace=True)

    return data


# get features of off-target models

# input columns: the 'target sequence' column represents 63bp wild-type sequence;

# the 'off-target sequence' column represents 63bp off-target sequence.


def off_target_predict(
    model_directory,
    rnn_params,
    model_func,
    fixed_feat,
    feat_label,
    off_model_path,
    data,
):

    # fixed_feat, feat_label = off_target_params_dict[cell]

    # features

    data = get_on_target_features(model_directory, rnn_params, model_func, data)

    x_data = off_target_mismatch_feature_engineering(data, fixed_feat, feat_label)

    x_data = np.array(x_data)

    # predict

    import joblib

    # off_model_path = off_model_path_dict[cell]

    model = joblib.load(off_model_path)

    ypred = model.predict(x_data)

    return ypred


def get_mismatch_pattern(target_seq, off_target_seq):

    pattern = ""

    for index, nucle in enumerate(target_seq):

        off_nucle = off_target_seq[index]

        if nucle == off_nucle:

            pattern = pattern + "-"

        else:

            pattern = pattern + "*"

    return pattern


# main


def main_off_predict(
    input_path,
    output_dir,
    on_model_directory,
    rnn_params,
    model_func,
    fixed_feat,
    feat_label,
    off_model_path,
):

    cell = "K562"

    mkdir(output_dir)

    output_path = output_dir + "/predicted_result_Aidit_OFF_%s.txt" % cell

    is_Exist_file(output_path)

    with open(output_path, "a") as a:

        wline = "target sequence\toff-target sequence\toff-target_score\n"

        a.write(wline)

        with open(input_path) as f:

            next(f)

            batch_n = 100000

            i = 0

            batch_data_dict = {"target sequence": [], "off-target sequence": []}

            for line in f:

                i += 1

                if i <= batch_n:

                    line = line.strip(" ").strip("\n")

                    wtseq, otseq = line.split("\t")

                    batch_data_dict["target sequence"].append(wtseq)

                    batch_data_dict["off-target sequence"].append(otseq)

                else:

                    # predict

                    batch_data = pd.DataFrame(batch_data_dict)

                    batch_ypred = off_target_predict(
                        on_model_directory,
                        rnn_params,
                        model_func,
                        fixed_feat,
                        feat_label,
                        off_model_path,
                        batch_data,
                    )

                    batch_data["off-target_score"] = batch_ypred

                    for index, row in batch_data.iterrows():

                        wline = "%s\t%s\t%s\n" % (
                            row["target sequence"],
                            row["off-target sequence"],
                            row["off-target_score"],
                        )

                        a.write(wline)

                    # initial

                    i = 0

                    batch_data_dict = {"target sequence": [], "off-target sequence": []}

            # last predict

            batch_data = pd.DataFrame(batch_data_dict)

            batch_ypred = off_target_predict(
                on_model_directory,
                rnn_params,
                model_func,
                fixed_feat,
                feat_label,
                off_model_path,
                batch_data,
            )

            batch_data["off-target_score"] = batch_ypred

            for index, row in batch_data.iterrows():

                wline = "%s\t%s\t%s\n" % (
                    row["target sequence"],
                    row["off-target sequence"],
                    row["off-target_score"],
                )

                a.write(wline)

    # output format transformation

    data = pd.read_csv(output_path, sep="\t")

    data["Guide mRNA + PAM Sequence (23bp)"] = data["target sequence"].apply(
        lambda x: x[20:43]
    )

    data["Potential Off Target Site (23bp)"] = data["off-target sequence"].apply(
        lambda x: x[20:43]
    )

    data["Mismatch Pattern"] = data.apply(
        lambda row: get_mismatch_pattern(
            row["Guide mRNA + PAM Sequence (23bp)"],
            row["Potential Off Target Site (23bp)"],
        ),
        axis=1,
    )

    data["Mismatch Number"] = data["Mismatch Pattern"].apply(lambda x: x.count("*"))

    data = data[
        [
            "target sequence",
            "off-target sequence",
            "Guide mRNA + PAM Sequence (23bp)",
            "Potential Off Target Site (23bp)",
            "Mismatch Pattern",
            "Mismatch Number",
            "off-target_score",
        ]
    ]

    data.rename(
        columns={
            "target sequence": "Target Sequence",
            "off-target sequence": "Off-target Sequence",
            "Guide mRNA + PAM Sequence (23bp)": "Guide mRNA + PAM Sequence (23 bp)",
            "Potential Off Target Site (23bp)": "Potential Off Target Site (23 bp)",
            "off-target_score": "Score",
        },
        inplace=True,
    )

    del data["Target Sequence"]

    del data["Off-target Sequence"]

    csv_output_path = output_dir + "/predicted_result_Aidit_OFF_%s.csv" % cell

    data.sort_values(by="Score", ascending=False, inplace=True)  # add

    data.to_csv(csv_output_path, index=False)

    is_Exist_file(output_path)


# SpCas9诱导的DNA双链断裂修复结果预测

# 3.1、同源修复特征工程

import pandas as pd

import numpy as np

import os

import warnings

warnings.filterwarnings("ignore")


# 基础功能 1：删除文件和创建文件夹

# 检查文件是否存在，存在删除


def is_Exist_file(path):

    import os

    import shutil

    if os.path.exists(path):

        try:

            shutil.rmtree(path)

        except NotADirectoryError as e:

            os.remove(path)


def mkdir(path):

    import os

    path = path.strip()

    path = path.rstrip("\\")

    isExists = os.path.exists(path)

    if not isExists:

        os.makedirs(path)


# 获取单位置核苷酸特征

# 基础功能 2：  Get sequence Feature

# get_dummies: feature one-hot encoding


def Get_Dummies(df, feature_list):

    df_dummies = pd.get_dummies(df[feature_list], columns=feature_list, prefix_sep="-")

    ## 去除 df 中含有 df_dummies 的列

    for col in df_dummies.columns.tolist():

        if col in df.columns.tolist():

            del df[col]

    ## concat

    df = pd.concat([df, df_dummies], axis=1)

    return df


def helper_single_feature_list(raw_data, seq_bp):

    raw_data["gRNAUp"] = raw_data["target sequence"].apply(lambda x: x[:20])

    raw_data["gRNATarget"] = raw_data["target sequence"].apply(lambda x: x[20:40])

    raw_data["PAM"] = raw_data["target sequence"].apply(lambda x: x[40:43])

    raw_data["gRNADown"] = raw_data["target sequence"].apply(lambda x: x[43:63])

    # 单位置核苷酸特征

    single_feature_list = []

    if seq_bp == 63:

        # Up

        for i in range(20):

            raw_data["S-U%s" % (i + 1)] = raw_data["gRNAUp"].apply(lambda x: x[i])

            single_feature_list.append("S-U%s" % (i + 1))

        # Target

        for i in range(20):

            raw_data["S-T%s" % (i + 1)] = raw_data["gRNATarget"].apply(lambda x: x[i])

            single_feature_list.append("S-T%s" % (i + 1))

        # PAM

        raw_data["S-PAM(N)"] = raw_data["PAM"].apply(lambda x: x[0])

        single_feature_list.append("S-PAM(N)")

        # Down

        for i in range(20):

            raw_data["S-D(-%s)" % (i + 1)] = raw_data["gRNADown"].apply(lambda x: x[i])

            single_feature_list.append("S-D(-%s)" % (i + 1))

    else:  # 28bp

        # Target

        for i in range(20):

            raw_data["S-T%s" % (i + 1)] = raw_data["gRNATarget"].apply(lambda x: x[i])

            single_feature_list.append("S-T%s" % (i + 1))

        # PAM

        raw_data["S-PAM(N)"] = raw_data["PAM"].apply(lambda x: x[0])

        single_feature_list.append("S-PAM(N)")

        # Down

        for i in range(5):

            raw_data["S-D(-%s)" % (i + 1)] = raw_data["gRNADown"].apply(lambda x: x[i])

            single_feature_list.append("S-D(-%s)" % (i + 1))

    del raw_data["gRNAUp"]

    del raw_data["gRNATarget"]

    del raw_data["PAM"]

    del raw_data["gRNADown"]

    return single_feature_list


def obtain_single_sequence_one_hot_feature_2nd(data, seq_bp):

    import time

    import copy

    raw_data = copy.deepcopy(data)

    print("================================")

    print("Function: Obtain_Single_Sequence_One_Hot_Feature ... ...")

    s = time.time()

    single_feature_list = helper_single_feature_list(raw_data, seq_bp)

    raw_data = Get_Dummies(raw_data, single_feature_list)

    # check all one-hot features in raw_data.columns & complement

    import copy

    single_one_hot_feature_list = []

    nfeat_list = copy.deepcopy(raw_data.columns.tolist())

    for feat in single_feature_list:

        for nucle in ["A", "C", "G", "T"]:

            one_hot_feat = feat + "-" + nucle

            single_one_hot_feature_list.append(one_hot_feat)

            if one_hot_feat not in nfeat_list:  # 补充不完整的 one-hot 特征

                raw_data[one_hot_feat] = 0

        del raw_data[feat]  # 删除非 one-hot 过渡特征

    raw_data = raw_data[["target sequence"] + single_one_hot_feature_list]

    e = time.time()

    print("Using Time: %s" % (e - s))

    print("================================\n")

    return raw_data


# 获取微同源特征

# 基础功能 3：  Get MH Feature

# Deletion Classes


def deletion_classes(edit_sites, max_len=30):

    min_site = min(edit_sites)

    max_site = max(edit_sites)

    delt_classes = []

    for i in range(1, max_len):

        inf_site = min_site - i + 1

        for site in range(inf_site, max_site + 1):

            key = "%s:%sD-%s" % (site, site + i - 1, i)

            delt_classes.append(key)

    delt_classes.append("D%s+" % (max_len))

    return delt_classes


# get MH feature

# Get 1bp MH


def help_1bp_MH(delt_inf_site, delt_sup_site, gRNASeq):

    delt_nucle = gRNASeq[delt_inf_site]

    MH_nucle = gRNASeq[delt_sup_site + 1]

    if delt_nucle == MH_nucle:

        MH = 1

    else:

        MH = 0

    return MH


# Get 2bp MH


def help_2bp_MH(delt_inf_site, delt_sup_site, gRNASeq):

    delt_nucles = gRNASeq[delt_inf_site : (delt_inf_site + 2)]

    MH_nucles = gRNASeq[(delt_sup_site + 1) : (delt_sup_site + 3)]

    if delt_nucles == MH_nucles:

        MH = 1

    else:

        MH = 0

    return MH


# Get 3bp MH


def help_3bp_MH(delt_inf_site, delt_sup_site, gRNASeq):

    delt_nucles = gRNASeq[delt_inf_site : (delt_inf_site + 3)]

    MH_nucles = gRNASeq[(delt_sup_site + 1) : (delt_sup_site + 4)]

    if delt_nucles == MH_nucles:

        MH = 1

    else:

        MH = 0

    return MH


# Get 4bp MH


def help_4bp_MH(delt_inf_site, delt_sup_site, gRNASeq):

    delt_nucles = gRNASeq[delt_inf_site : (delt_inf_site + 4)]

    MH_nucles = gRNASeq[(delt_sup_site + 1) : (delt_sup_site + 5)]

    if delt_nucles == MH_nucles:

        MH = 1

    else:

        MH = 0

    return MH


# MH: 1bp, 2bp, 3bp

# get MH feature


def Get_MH_Feature(gRNASeq, delt_classes, max_len=30):

    MH_feat_dict = {}

    for one_class in delt_classes:

        if one_class != "D%s+" % (max_len):

            delt_len = int(one_class.split("-")[1])

            delt_p = one_class.split("-")[0]

            delt_inf_site = int(delt_p.split(":")[0]) - 1

            delt_sup_site = int(delt_p.split(":")[1][:-1]) - 1

            if delt_len == 1:

                MH = help_1bp_MH(delt_inf_site, delt_sup_site, gRNASeq)

                MH_feat_dict["%s_1bp" % (one_class)] = MH

            elif delt_len == 2:

                MH1 = help_1bp_MH(delt_inf_site, delt_sup_site, gRNASeq)

                MH2 = help_2bp_MH(delt_inf_site, delt_sup_site, gRNASeq)

                MH_feat_dict["%s_1bp" % (one_class)] = MH1

                MH_feat_dict["%s_2bp" % (one_class)] = MH2

            elif delt_len == 3:

                MH1 = help_1bp_MH(delt_inf_site, delt_sup_site, gRNASeq)

                MH2 = help_2bp_MH(delt_inf_site, delt_sup_site, gRNASeq)

                MH3 = help_3bp_MH(delt_inf_site, delt_sup_site, gRNASeq)

                MH_feat_dict["%s_1bp" % (one_class)] = MH1

                MH_feat_dict["%s_2bp" % (one_class)] = MH2

                MH_feat_dict["%s_3bp" % (one_class)] = MH3

            else:

                # try:

                MH1 = help_1bp_MH(delt_inf_site, delt_sup_site, gRNASeq)

                # except IndexError as e:

                #     print("\ngRNASeq: ", gRNASeq)

                #     print("one_class: ", one_class)

                #     print("delt_inf_site: ", delt_inf_site)

                #     print("delt_sup_site: ", delt_sup_site)

                #     raise e

                MH2 = help_2bp_MH(delt_inf_site, delt_sup_site, gRNASeq)

                MH3 = help_3bp_MH(delt_inf_site, delt_sup_site, gRNASeq)

                MH4 = help_4bp_MH(delt_inf_site, delt_sup_site, gRNASeq)

                MH_feat_dict["%s_1bp" % (one_class)] = MH1

                MH_feat_dict["%s_2bp" % (one_class)] = MH2

                MH_feat_dict["%s_3bp" % (one_class)] = MH3

                MH_feat_dict["%s_4bp" % (one_class)] = MH4

        else:

            pass

    # sorting

    keys = list(MH_feat_dict.keys())

    keys.sort(reverse=False)

    MH_feats = [MH_feat_dict[key] for key in keys]

    return (MH_feats, keys)


# adjust two gRNASeq_85bp


def adjust_column_gRNASeq_85bp(gRNA_name, gRNASeq_85bp):

    if (gRNA_name != "AC026748.1_5_ENST00000624349_ATCAGCGCTGAGCCCATCAG") & (
        gRNA_name != "AC026748.1_4_ENST00000624349_AGCGCTGATGGGCTCAGCGC"
    ):

        return gRNASeq_85bp

    else:

        if (gRNA_name == "AC026748.1_5_ENST00000624349_ATCAGCGCTGAGCCCATCAG") & (
            gRNASeq_85bp is np.nan
        ):

            return "AGCTATAGGTCCAAGAGCCCATCAGCGCTGAGCCCATCAGCGCTGAGCCCATCAGGGGCTCGTGTCCTGGGCCTCCGGGTTTGTATTACCGCCATGCATT"

        elif (gRNA_name == "AC026748.1_4_ENST00000624349_AGCGCTGATGGGCTCAGCGC") & (
            gRNASeq_85bp is np.nan
        ):

            return "AGCTATAGGTCCAAGGGCTCAGCGCTGATGGGCTCAGCGCTGATGGGCTCAGCGCTGGGCTTGAGAGCAGGAGTGTGTGTTTGTATTACCGCCATGCATT"

        else:

            return gRNASeq_85bp


def assertion(seq):

    assert seq[41:43] == "GG"

    return seq


# 主函数: Get MH feature


def main_MH_Feature_2nd(data, edit_sites, max_len=30):

    import copy

    df = copy.deepcopy(data)

    # adjust two gRNASeq_85bp

    # df['target sequence'] = df.apply(lambda row: adjust_column_gRNASeq_85bp(row['sgRNA_name'], row['target sequence']),

    #                               axis=1)

    df["target sequence"] = df["target sequence"].apply(lambda x: assertion(x))

    # get mh features

    delt_classes = deletion_classes(edit_sites, max_len)

    df["MH_features"] = df["target sequence"].apply(
        lambda x: Get_MH_Feature(x, delt_classes, max_len)[0]
    )

    MH_data = pd.DataFrame(list(np.array(df["MH_features"])))

    # get columns

    gRNASeq_85bp = "AGCTATAGGTCCAAGAGCCCATCAGCGCTGAGCCCATCAGCGCTGAGCCCATCAGGGGCTCGTGTCCTGGGCCTCCGGGTTTGTATTACCGCCATGCATT"

    cols = Get_MH_Feature(gRNASeq_85bp, delt_classes, max_len)[1]

    MH_data.columns = cols

    del df["MH_features"]

    df = pd.concat([df[["target sequence"]], MH_data], axis=1)

    return df


# 3.2、SpCas9-induced DSB修复结果预测

import sys

from DSB_repair_features import *


# Get train & test data


def Obtain_predicting_feature_2nd(data, seq_bp=28, max_len=30):

    # 1. to get sequence feature

    seq_data = obtain_single_sequence_one_hot_feature_2nd(data, seq_bp)

    # 2. to get MH feature

    edit_sites = [34, 35, 36, 37, 38, 39, 40]

    MH_data = main_MH_Feature_2nd(data, edit_sites, max_len)

    return (seq_data, MH_data)


# For XGBoost Ensamble


def xgb_prediction(Xdata, model_path):

    import joblib

    model = joblib.load(model_path)  # 加载

    ypred = model.predict(Xdata)

    return ypred


def main_xgb_prediction(int_data, data, model_path_pattern, seq_bp=63, max_len=30):

    # Get Xdata

    seq_data, MH_data = Obtain_predicting_feature_2nd(data, seq_bp, max_len)

    Xdata = pd.merge(seq_data, MH_data, how="inner", on=["target sequence"])

    del Xdata["target sequence"]

    Xdata = np.array(Xdata)

    print("----------------------")

    print("Xtrain.shape:", Xdata.shape)

    # Get Engineered Feature

    # model_path = "XGB_K562-63bp_%s.model"%("29:40D-12")

    eng_data = seq_data[["target sequence"]]

    for model_label in int_data["new category"].unique():

        model_label = model_label.replace(":", "_")

        temp_model_path = model_path_pattern % (model_label)

        ypred = xgb_prediction(Xdata, temp_model_path)

        eng_data[model_label] = ypred

    return (seq_data, MH_data, eng_data)


# prediction

# 自定义损失函数


def my_categorical_crossentropy_2(labels, logits):

    import tensorflow as tf

    """ 

    label = tf.constant([[0,0,1,0,0]], dtype=tf.float32) 

    logit = tf.constant([[-1.2, 2.3, 4.1, 3.0, 1.4]], dtype=tf.float32) 

    logits = tf.nn.softmax(logit) # 计算softmax 

    my_result1 = my_categorical_cross_entropy(labels=label, logits=logits) 

    my_result2 = my_categorical_crossentropy_1(label, logits) 

    my_result3 = my_categorical_crossentropy_2(label, logits) 

    my_result1, my_result2, my_result3 

    """

    return tf.keras.losses.categorical_crossentropy(labels, logits)


def prediction(model_path, Xdata):

    # load model

    from keras.models import load_model

    model = load_model(
        model_path,
        custom_objects={"my_categorical_crossentropy_2": my_categorical_crossentropy_2},
    )

    # prediction & evluation

    ypred = model.predict(Xdata)

    return ypred


def batch_predict(batch_data, int_data, xgb_model_path, ensamble_model_path):

    seq_data, MH_data, eng_data = main_xgb_prediction(
        int_data, batch_data, xgb_model_path, seq_bp=63
    )

    feat_list = [seq_data, MH_data, eng_data]

    Xdata = pd.concat([temp_data.iloc[:, 1:] for temp_data in feat_list], axis=1)

    Xdata = np.array(Xdata)

    # prediction

    ypred = prediction(ensamble_model_path, Xdata)

    ypred = pd.DataFrame(ypred)

    ypred.columns = list(int_data["new category"].unique())

    ypred = pd.concat([seq_data[["target sequence"]], ypred], axis=1)

    return ypred


def write_content(data, data_path):

    cols = data.columns.tolist()

    if os.path.exists(data_path):

        line_format = "%s" + "\t%s" * (data.shape[0] - 1) + "\n"

        with open(data_path, "a") as a:

            for index, row in data.iterrows():

                line = line_format % tuple([row[col] for col in cols])

                a.write(line)

    else:

        data.to_csv(data_path, sep="\t", index=False)


# for ensemble prediction


def main_dsb_repair_predict(
    cell, input_path, output_dir, int_data_path, xgb_model_path, ensamble_model_path
):

    # parameters

    # int_data_path, xgb_model_path, ensamble_model_path = DSB_model_params_dict[cell]

    int_data = pd.read_csv(int_data_path)

    # read

    mkdir(output_dir)

    output_path = output_dir + "/predicted_result_Aidit_DSB_%s.txt" % cell

    is_Exist_file(output_path)

    with open(input_path, "r") as f:

        next(f)

        batch_n = 100000

        i = 0

        batch_data_dict = {"target sequence": []}

        for line in f:

            i += 1

            if i <= batch_n:

                line = line.strip(" ").strip("\n")

                wtseq = line.split("\t")[0]

                batch_data_dict["target sequence"].append(wtseq)

            else:

                # predict

                batch_data = pd.DataFrame(batch_data_dict)

                ypred = batch_predict(
                    batch_data, int_data, xgb_model_path, ensamble_model_path
                )

                write_content(ypred, output_path)

                # initial

                i = 0

                batch_data_dict = {"target sequence": []}

        # predict

        batch_data = pd.DataFrame(batch_data_dict)

        ypred = batch_predict(batch_data, int_data, xgb_model_path, ensamble_model_path)

        write_content(ypred, output_path)


# 日志记录

import logging


def Logger(logfile):

    # 第一步，创建一个logger

    logger = logging.getLogger()

    logger.setLevel(logging.DEBUG)  # Log等级开关

    # 第二步，创建一个handler，用于写入日志文件

    file_handler = logging.FileHandler(logfile, mode="w")

    file_handler.setLevel(logging.ERROR)  # 输出到file的log等级的开关

    # 第三步，定义handler的输出格式

    formatter = logging.Formatter(
        "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
    )

    file_handler.setFormatter(formatter)

    # 第四步，将handler添加到logger里面

    logger.addHandler(file_handler)

    # 如果需要同時需要在終端上輸出，定義一個streamHandler

    print_handler = logging.StreamHandler()  # 往屏幕上输出

    print_handler.setFormatter(formatter)  # 设置屏幕上显示的格式

    logger.addHandler(print_handler)

    return logger


# 预测主文件

# 5.1、多种on-target 预测输入

import os

import pandas as pd

import warnings

warnings.filterwarnings("ignore")


def is_Exist_file(path):

    import os

    import shutil

    if os.path.exists(path):

        try:

            shutil.rmtree(path)

        except NotADirectoryError as e:

            os.remove(path)


def mkdir(path):

    import os

    path = path.strip()

    path = path.rstrip("\\")

    isExists = os.path.exists(path)

    if not isExists:

        os.makedirs(path)

    else:

        pass


def walk(path):

    import os

    input_path_list = []

    if not os.path.exists(path):

        return -1

    for root, dirs, names in os.walk(path):

        for filename in names:

            input_path = os.path.join(root, filename)

            input_path_list.append(input_path)

    return input_path_list


# reverse complement


def reverse_complement(seq):

    rev_seq = ""

    rev_dict = {
        "A": "T",
        "a": "t",
        "C": "G",
        "c": "g",
        "G": "C",
        "g": "c",
        "T": "A",
        "t": "a",
    }

    for index, nucle in enumerate(seq):

        rev_nucle = rev_dict[nucle]

        rev_seq = rev_nucle + rev_seq

    return rev_seq


# find PAM


def find_pam(seq, pattern):

    seq = seq.upper()

    uplen = 0

    index_list = []

    try:

        while True:

            temp_seq = seq[uplen:]

            index = temp_seq.index(pattern)

            index = uplen + index

            index_list.append(index)

            uplen = index + 1

    except ValueError as e:

        pass

    return index_list


def location_prediction_site(index, strand, sep_length, seq_len):

    if strand == "+":

        return index - 41

    else:

        return sep_length - (index + 22 + (seq_len - 63))


# DataFrame


def get_sequence_data_2nd(seq, pattern, seq_len):

    data_dict = {"gRNASeq": [], "seq_len": [], "pam_index": [], "index": []}

    index_list = find_pam(seq, pattern)

    seq = seq.upper()

    for index in index_list:

        dindex = len(seq) - (index + 2)

        if (index < 41) & (dindex < 20 + (seq_len - 63)):

            data_dict["gRNASeq"].append("X" * (41 - index) + seq + "X" * (20 - dindex))

            data_dict["seq_len"].append(len(seq))

        elif (index < 41) & (dindex >= 20 + (seq_len - 63)):

            data_dict["gRNASeq"].append(
                "X" * (41 - index) + seq[: (index + 22 + (seq_len - 63))]
            )

            data_dict["seq_len"].append(len(seq[: (index + 22 + (seq_len - 63))]))

        elif (index >= 41) & (dindex < 20 + (seq_len - 63)):

            data_dict["gRNASeq"].append(seq[(index - 41) :] + "X" * (20 - dindex))

            data_dict["seq_len"].append(len(seq[(index - 41) :]))

        else:

            data_dict["gRNASeq"].append(
                seq[(index - 41) : (index + 22 + (seq_len - 63))]
            )

            data_dict["seq_len"].append(
                len(seq[(index - 41) : (index + 22 + (seq_len - 63))])
            )

        data_dict["pam_index"].append(40)

        data_dict["index"].append(index)

    # DataFrame

    data = pd.DataFrame(data_dict)

    return data


# input fasta file


def parse_sequence_2nd(seq, pattern, seq_len=63):

    rev_seq = reverse_complement(seq)

    data1 = get_sequence_data_2nd(seq, pattern, seq_len)

    data2 = get_sequence_data_2nd(rev_seq, pattern, seq_len)

    data1["strand"] = "+"

    data2["strand"] = "-"

    data = pd.concat([data1, data2], axis=0)

    # data = data.loc[data['seq_len'] == seq_len, :]

    data.reset_index(drop=True, inplace=True)

    data.rename(columns={"gRNASeq": "gRNASeq_%sbp" % seq_len}, inplace=True)

    data["upstream"] = data["gRNASeq_%sbp" % seq_len].apply(lambda x: x[:20])

    data["target"] = data["gRNASeq_%sbp" % seq_len].apply(lambda x: x[20:40])

    data["PAM"] = data["gRNASeq_%sbp" % seq_len].apply(lambda x: x[40:43])

    data["downstream"] = data["gRNASeq_%sbp" % seq_len].apply(lambda x: x[43:63])

    data["ddownstream"] = data["gRNASeq_%sbp" % seq_len].apply(lambda x: x[63:])

    data = data[
        ["upstream", "target", "PAM", "downstream", "ddownstream", "strand", "index"]
    ]

    sep_length = len(seq)

    data["index"] = data.apply(
        lambda row: location_prediction_site(
            row["index"], row["strand"], sep_length, seq_len
        ),
        axis=1,
    )

    return data


# For FASTA file input


def format_input_sequence_2nd(data, seq_label, seq_len, rcol="gRNASeq_63bp"):

    if seq_label == "All":

        isite = 0

        ssite = 63

    elif seq_label == "Both":

        isite = (63 - seq_len) / 2

        ssite = isite + seq_len

    elif seq_label == "Down":

        isite = 20

        ssite = isite + seq_len

    elif seq_label == "Up":

        isite = 20 - (seq_len - 23)

        ssite = 43

    else:

        raise (
            "parameter 'seq_label' not in ['Both', 'Down', 'Up'], is %s. Please check and try again."
            % (seq_label)
        )

    data["index"] = data["index"].apply(lambda x: x + int(isite))

    data["target sequence"] = data[rcol].apply(lambda x: x[int(isite) : int(ssite)])

    return data


def add_sequence(row):

    return (
        row["upstream"]
        + row["target"]
        + row["PAM"]
        + row["downstream"]
        + row["ddownstream"]
    )


def count_ATGC(seq):

    return seq.count("A") + seq.count("T") + seq.count("G") + seq.count("C")


def parse_fasta_file_2nd(fasta_file_path, seq_label, seq_len, logger):
    """

    parameters:

    fasta_file_path: a FASTA file path

    (seq_label, seq_len): [(all, 63), (Both, 23, 33, 43, 53), (Down, 28), (Up, 28)]

    for selection of on-target models

    return:

    DataFrame containing columns: ['ID', "target sequence", 'Location', 'Strand',

                                   'sgRNA Sequence (20bp)', 'GC Contents (%)']

    """

    data = pd.DataFrame()

    with open(fasta_file_path, "r") as f:

        s = 0

        for line in f:

            if (s == 0) & (line[0] == ">"):

                seq_id = line.strip("\n").strip(" ")[1:]

            elif ((s % 2) == 1) & (line[0] != ">"):

                seq = line.strip("\n").strip(" ")

                data0 = parse_sequence_2nd(seq, "GG")

                data0["ID"] = seq_id

                data = pd.concat([data, data0], axis=0)

            elif ((s % 2) == 0) & (line[0] == ">"):

                seq_id = line.strip("\n").strip(" ")[1:]

            else:

                logger.error(
                    "ValueError: There is a problem with the input FASTA format. Please check and try again."
                )

                raise ValueError(
                    "There is a problem with the input FASTA format. Please check and try again."
                )

            s += 1

    data.reset_index(drop=True, inplace=True)

    if data.shape[0] != 0:

        data["gRNASeq"] = data.apply(lambda row: add_sequence(row), axis=1)

        data = format_input_sequence_2nd(data, seq_label, seq_len, rcol="gRNASeq")

        data["sgRNA Sequence (20bp)"] = data["target"]

        data["GC Contents (%)"] = data["sgRNA Sequence (20bp)"].apply(
            lambda x: int((x.count("G") + x.count("C")) * 100 / 20)
        )

        data.rename(columns={"strand": "Strand"}, inplace=True)

        data["Location"] = data["index"].apply(
            lambda x: (
                "%s-%s" % (x + 1, x + seq_len) if x >= 0 else "%s-%s" % (0, x + seq_len)
            )
        )

        data = data[
            [
                "ID",
                "target sequence",
                "Location",
                "Strand",
                "sgRNA Sequence (20bp)",
                "GC Contents (%)",
            ]
        ]

        data["ATGC-count"] = data["target sequence"].apply(lambda x: count_ATGC(x))

        data = data.loc[data["ATGC-count"] == seq_len, :]

        del data["ATGC-count"]

        data.reset_index(drop=True, inplace=True)

        if data.shape[0] != 0:

            return data

        else:

            logger.error(
                "ValueError: Did not find a suitable 63bp-length prediction site for our algorithm. "
                "Please check and try again."
            )

            raise ValueError(
                "Did not find a suitable 63bp-length prediction site for our algorithm. Please check and try again."
            )

    else:

        logger.error(
            "ValueError: Did not find a suitable 63bp-length prediction site for our algorithm. "
            "Please check and try again."
        )

        raise ValueError(
            "Did not find a suitable 63bp-length prediction site for our algorithm. Please check and try again."
        )


# Gene Name Input

# relocation


def reLocation_2nd(row):

    location, index, seq = row["Location"], row["index"], row["target sequence"]

    seq_len = len(seq.strip("X"))

    if index < 0:

        sindex = 0

    else:

        sindex = index

    loc_p = location.split(":")

    if "(-)" in loc_p[1]:

        new_loc = int(loc_p[1].split(")")[1].split("-")[0]) + sindex

    else:

        new_loc = int(loc_p[1].split("-")[0]) + sindex

    return "%s:%s-%s" % (loc_p[0], str(new_loc), str(new_loc + seq_len - 1))


# check gene name


def check_gene_name_2nd(
    genome,
    gene_name,
    seq_label,
    seq_len,
    genome_ref_table_dict,
    gene_index_dir_dict,
    logger,
):

    ref_table_path = genome_ref_table_dict[genome]

    ref_table = pd.read_csv(ref_table_path, sep="\t")

    name_dict = {
        "Gene_name": ref_table["Gene_name"].unique(),
        "Gene_stable_ID": ref_table["Gene_stable_ID"].unique(),
        "Transcript_stable_ID": ref_table["Transcript_stable_ID"].unique(),
        "RefSeq_mRNA_ID": ref_table["RefSeq_mRNA_ID"].unique(),
    }

    name = ""

    for stable_id, values in name_dict.items():

        if gene_name in values:

            name = stable_id

        else:

            pass

    if name != "":

        df = ref_table.loc[ref_table[name] == gene_name, :]

        df.reset_index(drop=True, inplace=True)

        gene_id = df.loc[0, "Gene_name"]

        refseq_id = df.loc[0, "RefSeq_mRNA_ID"]

        gene_index = "%s-%s-gRNAs.csv" % (gene_id, refseq_id)

        gene_index_path = gene_index_dir_dict[genome] + "/" + gene_index

        if os.path.exists(gene_index_path):

            gene_df = pd.read_csv(gene_index_path)

            df = df.loc[
                (df["Gene_name"] == gene_id) & (df["RefSeq_mRNA_ID"] == refseq_id), :
            ]

            df.reset_index(drop=True, inplace=True)

            df = df[["Gene_name", "RefSeq_mRNA_ID", "Exon_number", "Location"]]

            gene_df["Gene_name"] = gene_id

            gene_df["RefSeq_mRNA_ID"] = refseq_id

            gene_df.rename(columns={"exon": "Exon_number"}, inplace=True)

            gene_df = pd.merge(
                df,
                gene_df,
                on=["Gene_name", "RefSeq_mRNA_ID", "Exon_number"],
                how="right",
            )

            gene_df.rename(
                columns={
                    "Gene_name": "Gene Name",
                    "RefSeq_mRNA_ID": "RefSeq mRNA ID",
                    "Exon_number": "Exon Order",
                    "strand": "Strand",
                },
                inplace=True,
            )

            if gene_df.shape[0] != 0:

                gene_df["gRNASeq"] = gene_df.apply(
                    lambda row: row["upstream"]
                    + row["target"]
                    + row["PAM"]
                    + row["downstream"],
                    axis=1,
                )

                gene_df = format_input_sequence_2nd(
                    gene_df, seq_label, seq_len, rcol="gRNASeq"
                )

                gene_df["sgRNA Sequence (20bp)"] = gene_df["target"]

                gene_df["GC Contents (%)"] = gene_df["sgRNA Sequence (20bp)"].apply(
                    lambda x: int((x.count("G") + x.count("C")) * 100 / 20)
                )

                gene_df["Location"] = gene_df.apply(
                    lambda row: reLocation_2nd(row), axis=1
                )

                gene_df = gene_df[
                    [
                        "Gene Name",
                        "RefSeq mRNA ID",
                        "Exon Order",
                        "target sequence",
                        "Location",
                        "Strand",
                        "sgRNA Sequence (20bp)",
                        "GC Contents (%)",
                    ]
                ]

                gene_df["ATGC-count"] = gene_df["target sequence"].apply(
                    lambda x: count_ATGC(x)
                )

                gene_df = gene_df.loc[gene_df["ATGC-count"] == seq_len, :]

                del gene_df["ATGC-count"]

                gene_df.reset_index(drop=True, inplace=True)

                if gene_df.shape[0] != 0:

                    return gene_df

                else:

                    logger.error(
                        "ValueError: "
                        "Did not find a suitable prediction site for our algorithm in the gene name [%s]. "
                        "Please try to input a entire FASTA file." % gene_name
                    )

                    raise ValueError(
                        "Did not find a suitable prediction site for our algorithm in the gene name [%s]. "
                        "Please try to input a entire FASTA file." % (gene_name)
                    )

            else:

                logger.error(
                    "ValueError: "
                    "Did not find a suitable prediction site for our algorithm in the gene name [%s]. "
                    "Please try to input a entire FASTA file." % (gene_name)
                )

                raise ValueError(
                    "Did not find a suitable prediction site for our algorithm in the gene name [%s]. "
                    "Please try to input a entire FASTA file." % (gene_name)
                )

        else:

            logger.error(
                "ValueError: "
                "Did not find a suitable prediction site for our algorithm in the gene name [%s]. "
                "Please try to input a entire FASTA file." % (gene_name)
            )

            raise ValueError(
                "Did not find a suitable prediction site for our algorithm in the gene name [%s]. "
                "Please try to input a entire FASTA file." % (gene_name)
            )

    else:

        logger.error(
            "ValueError: The gene name [%s] doesn't exist in our databases. Please check and try again."
            % (gene_name)
        )

        raise ValueError(
            "The gene name [%s] doesn't exist in our databases. Please check and try again."
            % (gene_name)
        )


# 5.2、预测文件

from CRISPR_Config import GenomeConfig, LOGGING

from logging_recorder import *

from AiditON_Input import *

from on_target_predict import *

from off_target_predict import *

from DSB_repair_predict import *

from parse_DSB_repair_outcomes import *

import os

import argparse

import pandas as pd

import warnings

warnings.filterwarnings("ignore")


# Get Gene name

GenomeConfig = GenomeConfig()

genome_ref_table_dict = GenomeConfig.genome_ref_table_dict

gene_index_dir_dict = GenomeConfig.gene_index_dir_dict

# limitation for DSB input num

beyound_num = GenomeConfig.beyound_num

# logging

LOGFile = LOGGING()

logfile = LOGFile.filepath

mkdir("/".join(logfile.split("/")[:-1]))

logger = Logger(logfile)


def main():

    parser = argparse.ArgumentParser()

    # Required parameters

    parser.add_argument(
        "--predictor",
        default="AiditON",
        type=str,
        required=True,
        help="Choose a prediction algorithm from AiditON, AiditOFF and AiditDSB.",
    )

    parser.add_argument(
        "--input_form",
        default="FASTAFile",
        type=str,
        required=True,
        help="The Input Form Should be one of ways following: "
        "'FASTAFile', 'GeneName' for ON repair prediction. "
        "'85bpFormat', '85bpFormatPath' for DSB repair outcomes prediction. "
        "'PairSequences' or 'PairFile' for OFF prediction.",
    )

    parser.add_argument(
        "--cell_type",
        default="K562",
        type=str,
        required=True,
        help="Enter a cell type to select a computational model based on corresponding datasets.",
    )

    parser.add_argument(
        "--enzyme",
        default="SpCas9",
        type=str,
        required=True,
        help="Select a CRISPR enzyme.",
    )

    parser.add_argument(
        "--AiditON_Input_Format",
        default="All_63",
        type=str,
        help="When the user uses Aidit_ON to predict the activity of sgRNA in K562 cell , "
        "we provide an input format with a variety of sequence lengths to adapt to the sequence input by the user."
        "'All_63': upstream 20 bp + target 20 bp + PAM 3 bp + downstream 20 bp;"
        "'Both_23': only target 20 bp + PAM 3 bp;"
        "'Both_33': upstream 5 bp + target 20 bp + PAM 3 bp + downstream 5 bp;"
        "'Down_28': target 20 bp + PAM 3 bp + downstream 5 bp;",
    )

    parser.add_argument(
        "--output_dir",
        default="../results/",
        type=str,
        help="The directory where the computational model will save results.",
    )

    parser.add_argument(
        "--fasta_file_path",
        default="../data/examples_sequences.fasta",
        type=str,
        help="Path to input sequences in FASTA form.",
    )

    parser.add_argument(
        "--genome",
        default="Homo_sapiens",
        type=str,
        help="To determine the selected genome in ['Homo_sapiens', 'Mus_musculus'].",
    )

    parser.add_argument(
        "--gene_name",
        default="A4GNT",
        type=str,
        help="Enter a gene name or gene stable ID or transcript stable ID or reference sequence mRNA ID.",
    )

    parser.add_argument(
        "--input_85bp_sequence",
        default="AGCCTCATAGATTCGATAGCATTCTGTGTTTACTCCGACCGGGCACAATGTGAGATCACTCTGGTTTGTATTACCGCCATGCATT",
        type=str,
        help="Sequence input for DSB repair prediction. The format of sequence is required:"
        "upstream 20bp + target 20bp + PAM 3bp + downstream 42bp.",
    )

    parser.add_argument(
        "--input_target_sequence_path",
        default="../data/examples_for_AiditDSB.txt",
        type=str,
        help="Path to input of target sequences for AiditDSB. Each sequence is required the 85bp sequence format",
    )

    parser.add_argument(
        "--on_seq",
        default="CCTAGAGTTGACTCCTGGATAAGCACACTAGGGAAAGCTATGGTCTGGTTACATACCACACAC",
        type=str,
        help="The target sequence input (upstream 20bp + target 20bp + PAM 3bp + downstream 20bp)"
        "for off-target prediction.",
    )

    parser.add_argument(
        "--off_seq",
        default="TATATAGGTGAGAGCAGACGTAAGCACATAGGGAAAGATAAGGACCAGTGCATTACCCAAAAG",
        type=str,
        help="The corresponding off-target sequence input (upstream 20bp + off_target 20bp + PAM 3bp + downstream 20bp)"
        " for off-target prediction.",
    )

    parser.add_argument(
        "--on_off_pair_input_path",
        default="../data/examples_on-off_sequence_pairs.txt",
        type=str,
        help="Path to On-Off target pair sequences for batch prediction. Columns "
        "['target sequence', 'off-target sequence'] in the file are separated by '\t'",
    )

    args = parser.parse_args()

    # input format for args.AiditON_Input_Format

    mkdir(args.output_dir)

    temp_input_path = args.output_dir + "/temp_input.txt"

    if args.AiditON_Input_Format in ["All_63", "Both_23", "Both_33", "Down_28"]:

        seq_label, seq_len = args.AiditON_Input_Format.split("_")

        seq_len = int(seq_len)

    else:

        logger.error(
            "ValueError: The parameter AiditON_Input_Format is not in "
            "['All_63', 'Both_23', 'Both_33', 'Down_28']. Please check and try again."
        )

        raise ValueError(
            "ValueError: The parameter AiditON_Input_Format is not in "
            "['All_63', 'Both_23', 'Both_33', 'Down_28']. Please check and try again."
        )

    if args.cell_type == "K562":

        pass

    else:

        if args.AiditON_Input_Format == "All_63":

            seq_label, seq_len = "All", 63

        else:

            logger.error(
                "ValueError: The input format of AiditON_%s does not contain parameter "
                "--AiditON_Input_Format %s. Please check and try again."
                % (args.cell_type, args.AiditON_Input_Format)
            )

            raise ValueError(
                "ValueError: The input format of AiditON_%s does not contain parameter "
                "--AiditON_Input_Format %s. Please check and try again."
                % (args.cell_type, args.AiditON_Input_Format)
            )

    # For input data

    if args.input_form == "FASTAFile":

        if os.path.exists(args.fasta_file_path):

            data = parse_fasta_file_2nd(
                args.fasta_file_path, seq_label, seq_len, logger
            )

            temp_data = data[["target sequence"]]

        else:

            logger.error(
                "ValueError: The input FASTA file doesn't exist. Please check and try again."
            )

            raise ValueError(
                "The input FASTA file doesn't exist. Please check and try again."
            )

    elif args.input_form == "GeneName":

        if args.genome in ["Homo_sapiens", "Mus_musculus"]:

            data = check_gene_name_2nd(
                args.genome,
                args.gene_name,
                seq_label,
                seq_len,
                genome_ref_table_dict,
                gene_index_dir_dict,
                logger,
            )

            temp_data = data[["target sequence"]]

        else:

            logger.error(
                "ValueError: The genome option is currently available in 'Homo sapiens' and 'Mus musculus'."
            )

            raise ValueError(
                "The genome option is currently available in 'Homo sapiens' and 'Mus musculus'."
            )

    elif args.input_form == "85bpFormat":

        temp_data = pd.DataFrame({"target sequence": [args.input_85bp_sequence]})

    elif args.input_form == "85bpFormatPath":

        temp_data = pd.read_csv(args.input_target_sequence_path)

        if temp_data.shape[0] > beyound_num:

            logger.error(
                "Beyound %s sequence: for batch mode please see our Github."
                % beyound_num
            )

            raise ValueError(
                "Beyound %s sequence: for batch mode please see our Github."
                % beyound_num
            )

    elif args.input_form == "PairSequences":

        on_seq = args.on_seq

        off_seq = args.off_seq

        temp_data = pd.DataFrame(
            {"target sequence": [on_seq], "off-target sequence": [off_seq]}
        )

    elif args.input_form == "PairFile":

        on_off_pair_input_path = args.on_off_pair_input_path

        temp_data = pd.read_csv(on_off_pair_input_path)

        temp_data.rename(
            columns={"potential off-target sequence": "off-target sequence"},
            inplace=True,
        )

    else:

        logger.error(
            "ValueError: The input_form option should be in "
            "['Enter FASTA text', 'Upload a FASTA file', 'Enter a gene name']."
        )

        raise ValueError(
            "The input_form option should be in ['Enter FASTA text', 'Upload a FASTA file', 'Enter a "
            "gene name']."
        )

    temp_data.to_csv(temp_input_path, sep="\t", index=False)

    # For Enzyme

    from CRISPR_Config import CRISPRAiditConfig

    # model parameters

    CRISPRConfig = CRISPRAiditConfig(logger, args.enzyme)

    # choose a computational model

    if args.predictor == "AiditON":

        if args.cell_type in ["K562", "Jurkat", "H1"]:

            pass

        else:

            logger.error(
                "ValueError: The cell_type Should be in [K562, Jurkat, H1] for AiditON, now is %s. "
                "Please check and "
                "try again." % args.cell_type
            )

            raise ValueError(
                "The cell_type Should be in [K562, Jurkat, H1] for AiditON, now is %s. "
                "Please check and try again." % args.cell_type
            )

        rnn_params = CRISPRConfig.rnn_params

        on_model_directory_dict = CRISPRConfig.on_model_directory_dict

        model_func = CRISPRConfig.model_function()

        on_model_directory = on_model_directory_dict[args.cell_type][
            args.AiditON_Input_Format
        ]

        main_on_predict(
            args.cell_type,
            on_model_directory,
            rnn_params,
            model_func,
            temp_input_path,
            args.output_dir,
            seq_len,
        )

        output_path = (
            args.output_dir + "/predicted_result_Aidit_ON_%s.csv" % args.cell_type
        )

        temp_data = pd.read_csv(output_path)

        data = pd.merge(data, temp_data, on=["target sequence"], how="inner")

        data["efficiency"] = data["efficiency"].apply(lambda x: x * 100)

        data.rename(
            columns={
                "target sequence": "Input Sequence (%s bp)" % seq_len,
                "efficiency": "efficiency (%)",
            },
            inplace=True,
        )

        data.to_csv(output_path, index=False)

        # data.columns for fasta file: ['ID', 'Input Sequence (%s bp)', 'Location', 'Strand',

        #                               'sgRNA Sequence (20bp)', 'GC Contents (%)', 'effciency']

        # data.columns for gene name: ["Gene Name", "RefSeq mRNA ID", "Exon Order", "Input Sequence (%s bp)",

        #                                      "Location", "Strand", "sgRNA Sequence (20bp)", "GC Contents (%)", "efficiency"]

        is_Exist_file(temp_input_path)

    elif args.predictor == "AiditOFF":

        if args.cell_type == "K562":

            pass

        else:

            logger.error(
                "ValueError: The cell_type Should be K562 for AiditOFF, now is %s. "
                "Please check and try again." % args.cell_type
            )

            raise ValueError(
                "The cell_type Should be K562 for AiditOFF, now is %s. Please check and try again."
                % args.cell_type
            )

        # check target sequence format

        check_data = pd.read_csv(temp_input_path, sep="\t")

        check_data["PAM"] = check_data["target sequence"].apply(lambda x: x[41:43])

        check_data["seq1_len"] = check_data["target sequence"].apply(lambda x: len(x))

        check_data["seq2_len"] = check_data["off-target sequence"].apply(
            lambda x: len(x)
        )

        seq1_len_dict = dict(check_data["seq1_len"].value_counts())

        seq2_len_dict = dict(check_data["seq2_len"].value_counts())

        pam_dict = dict(check_data["PAM"].value_counts())

        if (
            (pam_dict["GG"] == check_data.shape[0])
            & (seq1_len_dict[63] == check_data.shape[0])
            & (seq2_len_dict[63] == check_data.shape[0])
        ):

            on_model_directory_dict = CRISPRConfig.on_model_directory_dict

            on_model_directory = on_model_directory_dict[args.cell_type]["All_63"]

            rnn_params = CRISPRConfig.rnn_params

            model_func = CRISPRConfig.model_function()

            off_model_path_dict = CRISPRConfig.off_model_path_dict

            off_target_params_dict = CRISPRConfig.off_target_params_dict

            fixed_feat, feat_label = off_target_params_dict[args.cell_type]

            off_model_path = off_model_path_dict[args.cell_type]

            main_off_predict(
                temp_input_path,
                args.output_dir,
                on_model_directory,
                rnn_params,
                model_func,
                fixed_feat,
                feat_label,
                off_model_path,
            )

            is_Exist_file(temp_input_path)

        elif pam_dict["GG"] != check_data.shape[0]:

            is_Exist_file(temp_input_path)

            logger.error(
                "SequenceFormatError: there are the target sequences without an NGG PAM."
                "Please check and try again."
            )

            raise ValueError(
                "SequenceFormatError: there are the target sequences without an NGG PAM."
                "Please check and try again."
            )

        else:  # seq_len_dict[85] != check_data.shape[0]:

            is_Exist_file(temp_input_path)

            logger.error(
                "SequenceFormatError: there are the target sequences without 63bp length."
                "Please check and try again."
            )

            raise ValueError(
                "SequenceFormatError: there are the target sequences without 63bp length."
                "Please check and try again."
            )

    elif args.predictor == "AiditDSB":

        if args.cell_type in ["K562", "Jurkat"]:

            pass

        else:

            logger.error(
                "ValueError: The cell_type Should be in [K562, Jurkat] for AiditDSB, now is %s. "
                "Please check and try "
                "again." % args.cell_type
            )

            raise ValueError(
                "The cell_type Should be in [K562, Jurkat] for AiditDSB, now is %s. Please check and try "
                "again." % args.cell_type
            )

        # check target sequence format

        check_data = pd.read_csv(temp_input_path, sep="\t")

        check_data["target sequence"] = check_data["target sequence"].apply(
            lambda x: x + "GTTTGTATTACCGCCATGCATT"
        )  # add

        check_data.to_csv(temp_input_path, sep="\t", index=False)  # add

        check_data["seq_length"] = check_data["target sequence"].apply(lambda x: len(x))

        check_data["PAM"] = check_data["target sequence"].apply(lambda x: x[41:43])

        seq_len_dict = dict(check_data["seq_length"].value_counts())

        pam_dict = dict(check_data["PAM"].value_counts())

        if (pam_dict["GG"] == check_data.shape[0]) & (
            seq_len_dict[85] == check_data.shape[0]
        ):

            DSB_model_params_dict = CRISPRConfig.DSB_model_params_dict

            drop_cats_dict = CRISPRConfig.drop_cats_dict

            int_data_path, xgb_model_path, ensamble_model_path = DSB_model_params_dict[
                args.cell_type
            ]

            main_dsb_repair_predict(
                args.cell_type,
                temp_input_path,
                args.output_dir,
                int_data_path,
                xgb_model_path,
                ensamble_model_path,
            )

            is_Exist_file(temp_input_path)

            pred_input_path = (
                args.output_dir + "/predicted_result_Aidit_DSB_%s.txt" % args.cell_type
            )

            write_output_dir = args.output_dir + "/AiditDSB-%s/" % args.cell_type

            write_DSB_repair_outcomes(
                args.cell_type, pred_input_path, write_output_dir, drop_cats_dict
            )

        elif pam_dict["GG"] != check_data.shape[0]:

            is_Exist_file(temp_input_path)

            logger.error(
                "SequenceFormatError: there are the target sequences without an NGG PAM."
                "Please check and try again."
            )

            raise ValueError(
                "SequenceFormatError: there are the target sequences without an NGG PAM."
                "Please check and try again."
            )

        else:  # seq_len_dict[85] != check_data.shape[0]:

            is_Exist_file(temp_input_path)

            logger.error(
                "SequenceFormatError: there are the target sequences without 85bp length."
                "Please check and try again."
            )

            raise ValueError(
                "SequenceFormatError: there are the target sequences without 85bp length."
                "Please check and try again."
            )

    else:

        logger.error(
            "ValueError: The parameter predictor should be in  [AiditON, AiditOFF and AiditDSB], "
            "not %s. Please check and try again." % args.predictor
        )

        raise ValueError(
            "The parameter predictor should be in  [AiditON, AiditOFF and AiditDSB], "
            "not %s. Please check and try again." % args.predictor
        )

    print("Finish.")


# 5.3、输入参数设置


class CRISPRAiditConfig:

    def __init__(self, logger, enzyme="SpCas9", pred_dir="../data/predictions"):

        # ON-OFF

        self.logger = logger

        self.enzyme = enzyme

        self.pred_dir = pred_dir

        if self.enzyme == "SpCas9":

            self.rnn_params = {
                "bilstm_hidden1": 32,
                "bilstm_hidden": 64,
                "hidden1": 64,
                "dropout": 0.2276,
            }

            self.on_model_directory_dict = {
                "K562": {
                    "All_63": self.pred_dir
                    + "/on-target/k562/on-target_RNN-weights_for-K562",
                    "Both_23": self.pred_dir
                    + "/on-target/k562/Both_23bp/on-target_K562-23bp_model-2",
                    "Both_33": self.pred_dir
                    + "/on-target/k562/Both_33bp/on-target_K562-33bp_model-3",
                    "Down_28": self.pred_dir
                    + "/on-target/k562/Down_28bp/on-target_K562-28bp_model-1",
                },
                "Jurkat": {
                    "All_63": self.pred_dir
                    + "/on-target/jurkat/on-target_RNN-weights_for-Jurkat"
                },
                "H1": {
                    "All_63": self.pred_dir
                    + "/on-target/h1/on-target_RNN-weights_for-H1"
                },
            }

            self.off_model_path_dict = {
                "K562": self.pred_dir
                + "/off-target/off-target-best-model-for-K562.model"
            }

            self.off_target_params_dict = {"K562": ("all", "+P+M")}

            # DSB model parameters

            self.DSB_model_params_dict = {
                "K562": [
                    self.pred_dir
                    + "/DSB_repair/Integrated_K562_DSB_Repair_Merged_Information.csv",
                    self.pred_dir + "/DSB_repair/K562/XGB_K562-63bp_%s.model",
                    self.pred_dir
                    + "/DSB_repair/K562/DSB_repair-outcomes-ensamble-best-weight-for-K562.hdf5",
                ],
                "Jurkat": [
                    self.pred_dir
                    + "/DSB_repair/Integrated_Jurkat_DSB_Repair_Merged_Information.csv",
                    self.pred_dir + "/DSB_repair/Jurkat/XGB_Jurkat-63bp_%s.model",
                    self.pred_dir
                    + "/DSB_repair/Jurkat/DSB_repair-outcomes-ensamble-best-weight-for-Jurkat.hdf5",
                ],
            }

            self.drop_cats_dict = {
                "K562": ["D-17", "D-18+", "WildType"],
                "Jurkat": ["D-19+", "WildType"],
            }

        else:

            logger.error(
                "The parameter enzyme-%s does not exists. Please check and try again."
                % enzyme
            )

            raise ValueError(
                "The parameter enzyme-%s does not exists. Please check and try again."
                % enzyme
            )

    # custom evaluation function

    def get_spearman_rankcor(self, y_true, y_pred):

        from scipy.stats import spearmanr

        import tensorflow as tf

        return tf.py_function(
            spearmanr,
            [tf.cast(y_pred, tf.float32), tf.cast(y_true, tf.float32)],
            Tout=tf.float32,
        )

    # RNN

    # input 63bp length sequence

    def RNN(self, params, seq_len=63):

        from keras.models import Model

        from keras.layers import LSTM, Bidirectional

        from keras.layers import Input

        from keras.layers import Dense, Dropout

        # Model Frame

        visible = Input(shape=(seq_len, 4))

        bi_lstm1 = Bidirectional(
            LSTM(params["bilstm_hidden1"], dropout=0.2, return_sequences=True)
        )(visible)

        bi_lstm = Bidirectional(LSTM(params["bilstm_hidden"], dropout=0.2))(bi_lstm1)

        hidden1 = Dense(params["hidden1"], activation="relu")(bi_lstm)

        dropout = Dropout(params["dropout"])(hidden1)

        output = Dense(1)(dropout)

        # model architecture

        model = Model(inputs=visible, outputs=output)

        return model

    def model_function(self):

        if self.enzyme == "SpCas9":

            return self.RNN

        else:

            self.logger.error(
                "The parameter enzyme-%s does not exists. Please check and try again."
                % self.enzyme
            )

            raise ValueError(
                "The parameter enzyme-%s does not exists. Please check and try again."
                % self.enzyme
            )


class GenomeConfig:

    def __init__(self, genome_dir="../data", beyound_num=500):

        self.genome_ref_table_dict = {
            "Homo_sapiens": genome_dir + "/homo_sapiens/homo_sapiens.tsv",
            "Mus_musculus": genome_dir + "/mus_musculus/mus_musculus.tsv",
        }

        self.gene_index_dir_dict = {
            "Homo_sapiens": genome_dir + "/homo_sapiens/homo_sapiens-gRNAs",
            "Mus_musculus": genome_dir + "/mus_musculus/mus_musculus-gRNAs",
        }

        self.beyound_num = beyound_num


class LOGGING:

    def __init__(self, filepath="../results/log_record.txt"):

        self.filepath = filepath


# 第二章、模型训练代码

#  AIdit_Cas9_ON 模型训练

# 1.1、data.py

# -*-coding: utf-8 -*-

import os

import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings("ignore")


def is_Exist_file(path):

    import os

    if os.path.exists(path):

        os.remove(path)


def mkdir(path):

    import os

    path = path.strip()  # 去除首位空格

    path = path.rstrip("\\")  # 去除尾部 \ 符号

    isExists = os.path.exists(path)  # 判断路径是否存在

    # 判断结果

    if not isExists:

        os.makedirs(path)  # 如果不存在则创建目录

        print(path + " 创建成功")

    else:

        print(path + " 目录已存在")  # 如果目录存在则不创建，并提示目录已存在


## 需要遍历的目录树的路径

## 路径和文件名连接构成完整路径


def walk(path):

    import os

    input_path_list = []

    if not os.path.exists(path):

        return -1

    for root, dirs, names in os.walk(path):

        for filename in names:

            input_path = os.path.join(root, filename)

            input_path_list.append(input_path)

            # print(os.path.join(root,filename)) # 路径和文件名连接构成完整路径

    return input_path_list


# get best_checkpoint_path file


def get_best_checkpoint_path(path):

    file_list = walk(path)

    epoch_dict = {}

    for file in file_list:

        file_p = file.split("-")

        file_p = [s for s in file_p if len(s) != 0]

        epoch = int(file_p[-5])

        epoch_dict[epoch] = file

    epoch_max = max(list(epoch_dict.keys()))

    file_max = epoch_dict[epoch_max]

    return file_max


## 选择最大 epoch 的 5 个模型文件


def best_5_epoches_model(model_path_list, selected_epoch_n=5):

    model_epoch_dict = {}

    for model_path in model_path_list:

        model_file = model_path.split("/")[-1]

        model_count = int(model_file.split("-")[-5])

        model_epoch_dict[model_count] = model_path

    ## the highest 5

    model_epoch_list = list(model_epoch_dict.keys())

    model_epoch_list.sort(reverse=True)

    selected_epoches = model_epoch_list[:selected_epoch_n]

    selected_epoches.sort()

    selected_model_path_list = [
        model_epoch_dict[model_epoch] for model_epoch in selected_epoches
    ]

    return selected_model_path_list


## 深度学习输入1： 序列特征

# 生成 Seequence 数据


def find_all(sub, s):

    index = s.find(sub)

    feat_one = np.zeros(len(s))

    while index != -1:

        feat_one[index] = 1

        index = s.find(sub, index + 1)

    return feat_one


## 获取单样本序列数据


def obtain_each_seq_data(seq):

    A_array = find_all("A", seq)

    G_array = find_all("G", seq)

    C_array = find_all("C", seq)

    T_array = find_all("T", seq)

    one_sample = np.array([A_array, G_array, C_array, T_array])

    # print(one_sample.shape)

    return one_sample


## 获取序列数据

## 参数说明：

## data：输入的数据，要求含有gRNA列名，该列为原始序列值：target（20） + pam (3) + down (5)

## y: 输入的编辑效率list，对应 data

## layer: 为输出层的label， 用于调整输出数据的shape，可取值['1D', '2D']

## 输出：特征数据 x_data

# def obtain_Sequence_data(data, y, layer_label):


def obtain_Sequence_data(data, layer_label="1D"):

    x_data = []

    for i, row in data.iterrows():

        seq = row["gRNASeq"]

        one_sample = obtain_each_seq_data(seq)

        if layer_label == "1D":  # 用于LSTM or Conv1D, shape=(sample, step, feature)

            one_sample_T = one_sample.T

            x_data.append(one_sample_T)

        else:

            x_data.append(one_sample)

    x_data = np.array(x_data)

    # y = np.array(y)

    if layer_label == "2D":  # 用于 Conv2D shape=(sample, rows, cols, channels)

        x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], x_data.shape[2], 1)

        # y = y.reshape(y.shape[0], 1)

        print("Conv2D: shape=(sample, rows, cols, channels)")

    else:

        print("LSTM or Conv1D: shape=(sample, step, feature)")

    # y = y.astype('float32')

    x_data = x_data.astype("float32")

    print("After transformation, x_data.shape:", x_data.shape)

    return x_data


## train & validation data


def TrainValidation_KFoldData(read_fold_data_path, KFold_n, val_fold_n):

    data_list = []

    for fold_n in range(1, 1 + KFold_n, 1):

        one_read_fold_data_path = read_fold_data_path % (KFold_n, fold_n)

        if fold_n == val_fold_n:

            val_data = pd.read_csv(one_read_fold_data_path, sep="\t")

        else:

            temp_data = pd.read_csv(one_read_fold_data_path, sep="\t")

            data_list.append(temp_data)

    train_data = pd.concat(data_list, axis=0)

    train_data.reset_index(drop=True, inplace=True)

    print("\nValidation Fold number:", val_fold_n)

    print(
        "Train data.shape:", train_data.shape, "Validation data.shape:", val_data.shape
    )

    # print("Finish.")

    return train_data, val_data


## get model data


def main_model_data(
    read_fold_data_path, KFold_n, val_fold_n, seq_len, ycol, layer_label="1D"
):

    train_data, val_data = TrainValidation_KFoldData(
        read_fold_data_path, KFold_n, val_fold_n
    )

    train_data.rename(
        columns={"gRNASeq_%sbp" % (seq_len): "gRNASeq", ycol: "regressor_target"},
        inplace=True,
    )

    val_data.rename(
        columns={"gRNASeq_%sbp" % (seq_len): "gRNASeq", ycol: "regressor_target"},
        inplace=True,
    )

    ## get model data

    ## for training data

    x_train = obtain_Sequence_data(train_data, layer_label)

    x_train = x_train.astype("float32")

    y_train = train_data["regressor_target"]

    y_train = np.array(y_train)

    y_train = y_train.astype("float32")

    ## for validation data

    x_val = obtain_Sequence_data(val_data, layer_label)

    x_val = x_val.astype("float32")

    y_val = val_data["regressor_target"]

    y_val = np.array(y_val)

    y_val = y_val.astype("float32")

    return (x_train, y_train, x_val, y_val)


# 1.2、models.py

# -*-coding: utf-8 -*-

import numpy as np

import warnings

warnings.filterwarnings("ignore")


# RNN 4 层全连接


def BiLSTM_Model(params, seq_len):

    from keras.models import Model

    from keras.layers import LSTM, Bidirectional

    from keras.layers import Input

    from keras.layers import Dense, Dropout

    # Model Frame

    visible = Input(shape=(seq_len, 4))

    bi_lstm1 = Bidirectional(
        LSTM(params["bilstm_hidden1"], dropout=0.2, return_sequences=True)
    )(visible)

    bi_lstm = Bidirectional(LSTM(params["bilstm_hidden"], dropout=0.2))(bi_lstm1)

    # 全连接层 1

    hidden1 = Dense(params["hidden1"], activation="relu")(bi_lstm)

    dropout1 = Dropout(params["dropout1"])(hidden1)

    # 全连接层 2

    hidden2 = Dense(params["hidden2"], activation="relu")(dropout1)

    dropout2 = Dropout(params["dropout2"])(hidden2)

    # 全连接层 3

    back1 = Dense(params["hidden3"], activation="relu")(dropout2)

    output = Dense(1)(back1)

    #

    model = Model(inputs=visible, outputs=output)

    return model


# one reference param

one_BiLSTM_params = {
    "bilstm_hidden1": 64,
    "bilstm_hidden": 64,
    "hidden1": 128,
    "hidden2": 128,
    "hidden3": 12,
    "dropout1": 0.3,
    "dropout2": 0.2,
    "batch_size": 128,
    "optimizer": "Adam",
}


# GRU


def BiGRU_Model(params, seq_len):

    from keras.models import Model

    from keras.layers import LSTM, Bidirectional, GRU

    from keras.layers import Input

    from keras.layers import Dense, Dropout

    # Model Frame

    visible = Input(shape=(seq_len, 4))

    gru1 = Bidirectional(
        GRU(params["bigru_hidden1"], dropout=0.2, return_sequences=True)
    )(visible)

    gru = Bidirectional(GRU(params["bigru_hidden"], dropout=0.2))(gru1)

    # 全连接层 1

    hidden1 = Dense(params["hidden1"], activation="relu")(gru)

    dropout1 = Dropout(params["dropout1"])(hidden1)

    # 全连接层 2

    hidden2 = Dense(params["hidden2"], activation="relu")(dropout1)

    dropout2 = Dropout(params["dropout2"])(hidden2)

    # 全连接层 3

    back1 = Dense(params["hidden3"], activation="relu")(dropout2)

    output = Dense(1)(back1)

    #

    model = Model(inputs=visible, outputs=output)

    return model


# one reference param

one_BiGRU_params = {
    "bigru_hidden1": 128,
    "bigru_hidden": 64,
    "hidden1": 128,
    "hidden2": 128,
    "hidden3": 12,
    "dropout1": 0.3,
    "dropout2": 0.2,
    "batch_size": 128,
    "optimizer": "Adam",
}


# Conv2D


def Conv2D_model(params, seq_len):

    from keras.models import Model

    from keras.layers import Input

    from keras.layers import Conv2D, MaxPooling2D

    from keras.layers import Dense, Dropout

    from keras.layers import Flatten

    visible = Input(shape=(4, seq_len, 1))

    conv2d_1 = Conv2D(
        params["filters"],
        params["kernel_size"],
        strides=1,
        padding="same",
        activation="relu",
    )(visible)

    maxpool_1 = MaxPooling2D(pool_size=(2, 2))(conv2d_1)  # 池化层

    # 2nd

    conv2d_2 = Conv2D(
        params["filters"], (2, 2), strides=1, padding="same", activation="relu"
    )(maxpool_1)

    maxpool_2 = MaxPooling2D(pool_size=(2, 2))(conv2d_2)  # 池化层

    # Flatten + FC

    flat = Flatten()(maxpool_2)

    hidden1 = Dense(params["hidden1"], activation="relu")(flat)

    dropout1 = Dropout(params["dropout1"])(hidden1)

    hidden2 = Dense(params["hidden2"], activation="relu")(dropout1)

    dropout2 = Dropout(params["dropout2"])(hidden2)

    hidden3 = Dense(params["hidden3"], activation="relu")(dropout2)

    output = Dense(1)(hidden3)

    model = Model(inputs=visible, outputs=output)

    return model


one_BiCNN_params = {
    "filters": 128,
    "kernel_size": 3,
    "hidden1": 512,
    "hidden2": 256,
    "hidden3": 128,
    "dropout1": 0.2,
    "dropout2": 0.2,
    "batch_size": 128,
    "optimizer": "Adam",
}


# RNN +  3 层全连接


def BiLSTM_Model3(params, seq_len):

    from keras.models import Model

    from keras.layers import LSTM, Bidirectional

    from keras.layers import Input

    from keras.layers import Dense, Dropout

    # Model Frame

    visible = Input(shape=(seq_len, 4))

    bi_lstm1 = Bidirectional(
        LSTM(params["bilstm_hidden1"], dropout=0.2, return_sequences=True)
    )(visible)

    bi_lstm = Bidirectional(LSTM(params["bilstm_hidden"], dropout=0.2))(bi_lstm1)

    # 全连接层 1

    hidden1 = Dense(params["hidden1"], activation="relu")(bi_lstm)

    # 全连接层 2

    dropout = Dropout(params["dropout"])(hidden1)

    # 全连接层 3

    back1 = Dense(params["hidden2"], activation="relu")(dropout)

    output = Dense(1)(back1)

    #

    model = Model(inputs=visible, outputs=output)

    return model


# one reference param

one_BiLSTM_params3 = {
    "bilstm_hidden1": 64,
    "bilstm_hidden": 64,
    "hidden1": 128,
    "hidden2": 12,
    "dropout": 0.2,
    "batch_size": 128,
    "optimizer": "Adam",
}


# GRU + 3 FC


def BiGRU_Model3(params, seq_len):

    from keras.models import Model

    from keras.layers import LSTM, Bidirectional, GRU

    from keras.layers import Input

    from keras.layers import Dense, Dropout

    # Model Frame

    visible = Input(shape=(seq_len, 4))

    gru1 = Bidirectional(
        GRU(params["bigru_hidden1"], dropout=0.2, return_sequences=True)
    )(visible)

    gru = Bidirectional(GRU(params["bigru_hidden"], dropout=0.2))(gru1)

    # 全连接层 1

    hidden1 = Dense(params["hidden1"], activation="relu")(gru)

    # 全连接层 2

    dropout = Dropout(params["dropout"])(hidden1)

    # 全连接层 3

    back1 = Dense(params["hidden2"], activation="relu")(dropout)

    output = Dense(1)(back1)

    #

    model = Model(inputs=visible, outputs=output)

    return model


# one reference param

one_BiGRU_params3 = {
    "bigru_hidden1": 64,
    "bigru_hidden": 64,
    "hidden1": 128,
    "hidden2": 12,
    "dropout": 0.2,
    "batch_size": 128,
    "optimizer": "Adam",
}


# Conv2D


def Conv2D_model3(params, seq_len):

    from keras.models import Model

    from keras.layers import Input

    from keras.layers import Conv2D, MaxPooling2D

    from keras.layers import Dense, Dropout

    from keras.layers import Flatten

    visible = Input(shape=(4, seq_len, 1))

    conv2d_1 = Conv2D(
        params["filters"],
        params["kernel_size"],
        strides=1,
        padding="same",
        activation="relu",
    )(visible)

    maxpool_1 = MaxPooling2D(pool_size=(2, 2))(conv2d_1)  # 池化层

    # 2nd

    conv2d_2 = Conv2D(
        params["filters"], (2, 2), strides=1, padding="same", activation="relu"
    )(maxpool_1)

    maxpool_2 = MaxPooling2D(pool_size=(2, 2))(conv2d_2)  # 池化层

    # Flatten + FC

    flat = Flatten()(maxpool_2)

    hidden1 = Dense(params["hidden1"], activation="relu")(flat)

    dropout1 = Dropout(params["dropout1"])(hidden1)

    hidden2 = Dense(params["hidden2"], activation="relu")(dropout1)

    dropout2 = Dropout(params["dropout1"])(hidden2)

    output = Dense(1)(dropout2)

    model = Model(inputs=visible, outputs=output)

    return model


one_BiCNN_params3 = {
    "filters": 128,
    "kernel_size": 3,
    "hidden1": 256,
    "hidden2": 128,
    "dropout1": 0.2,
    "dropout2": 0.2,
    "batch_size": 128,
    "optimizer": "Adam",
}


# RNN + 2 层全连接


def BiLSTM_Model2(params, seq_len):

    from keras.models import Model

    from keras.layers import LSTM, Bidirectional

    from keras.layers import Input

    from keras.layers import Dense, Dropout

    # Model Frame

    visible = Input(shape=(seq_len, 4))

    bi_lstm1 = Bidirectional(
        LSTM(params["bilstm_hidden1"], dropout=0.2, return_sequences=True)
    )(visible)

    bi_lstm = Bidirectional(LSTM(params["bilstm_hidden"], dropout=0.2))(bi_lstm1)

    # 全连接层 1

    hidden1 = Dense(params["hidden1"], activation="relu")(bi_lstm)

    # 全连接层 2

    dropout = Dropout(params["dropout"])(hidden1)

    output = Dense(1)(dropout)

    #

    model = Model(inputs=visible, outputs=output)

    return model


# one reference param

one_BiLSTM_params2_1 = {
    "bilstm_hidden1": 64,
    "bilstm_hidden": 64,
    "hidden1": 12,
    "dropout": 0.2,
    "batch_size": 128,
    "optimizer": "Adam",
}

one_BiLSTM_params2 = {
    "bilstm_hidden1": 32,
    "bilstm_hidden": 64,
    "hidden1": 64,
    "dropout": 0.2276,
    "batch_size": 128,
    "optimizer": "Nadam",
}


# GRU + 2FC


def BiGRU_Model2(params, seq_len):

    from keras.models import Model

    from keras.layers import LSTM, Bidirectional, GRU

    from keras.layers import Input

    from keras.layers import Dense, Dropout

    # Model Frame

    visible = Input(shape=(seq_len, 4))

    gru1 = Bidirectional(
        GRU(params["bigru_hidden1"], dropout=0.2, return_sequences=True)
    )(visible)

    gru = Bidirectional(GRU(params["bigru_hidden"], dropout=0.2))(gru1)

    # 全连接层 1

    hidden1 = Dense(params["hidden1"], activation="relu")(gru)

    # 全连接层 2

    back1 = Dropout(params["dropout"])(hidden1)

    output = Dense(1)(back1)

    ##

    model = Model(inputs=visible, outputs=output)

    return model


one_BiGRU_params2 = {
    "bigru_hidden1": 64,
    "bigru_hidden": 32,
    "hidden1": 128,
    "dropout": 0.4919,
    "batch_size": 128,
    "optimizer": "Nadam",
}


## Conv2D


def Conv2D_model2(params, seq_len):

    from keras.models import Model

    from keras.layers import Input

    from keras.layers import Conv2D, MaxPooling2D

    from keras.layers import Dense, Dropout

    from keras.layers import Flatten

    visible = Input(shape=(4, seq_len, 1))

    conv2d_1 = Conv2D(
        params["filters"],
        params["kernel_size"],
        strides=1,
        padding="same",
        activation="relu",
    )(visible)

    maxpool_1 = MaxPooling2D(pool_size=(2, 2))(conv2d_1)  # 池化层

    # 2nd

    conv2d_2 = Conv2D(
        params["filters"], (2, 2), strides=1, padding="same", activation="relu"
    )(maxpool_1)

    maxpool_2 = MaxPooling2D(pool_size=(2, 2))(conv2d_2)  # 池化层

    ## Flatten + FC

    flat = Flatten()(maxpool_2)

    hidden1 = Dense(params["hidden1"], activation="relu")(flat)

    dropout = Dropout(params["dropout"])(hidden1)

    output = Dense(1)(dropout)

    model = Model(inputs=visible, outputs=output)

    return model


one_BiCNN_params2 = {
    "filters": 128,
    "kernel_size": 3,
    "hidden1": 256,
    "dropout": 0.3,
    "batch_size": 512,
    "optimizer": "Adam",
}


## 选择模型参数


def selection_model_parameter_tuple(model_label, hidden_num):

    if (model_label == "BiLSTM") & (hidden_num == 4):

        return one_BiLSTM_params

    elif (model_label == "BiLSTM") & (hidden_num == 3):

        return one_BiLSTM_params3

    elif (model_label == "BiLSTM") & (hidden_num == 2):

        return one_BiLSTM_params2

    elif (model_label == "BiGRU") & (hidden_num == 4):

        return one_BiGRU_params

    elif (model_label == "BiGRU") & (hidden_num == 3):

        return one_BiGRU_params3

    elif (model_label == "BiGRU") & (hidden_num == 2):

        return one_BiGRU_params2

    elif (model_label == "BiCNN") & (hidden_num == 4):

        return one_BiCNN_params

    elif (model_label == "BiCNN") & (hidden_num == 3):

        return one_BiCNN_params3

    elif (model_label == "BiCNN") & (hidden_num == 2):

        return one_BiCNN_params2

    else:

        print(
            "Error (model: %s; hidden_num: %s), Please check and try again."
            % (model_label, hidden_num)
        )

        return {}


# 选择模型


def selecting_deeplearning(model_label, hidden_num):

    if model_label == "BiLSTM":

        if hidden_num == 4:

            return BiLSTM_Model

        elif hidden_num == 3:

            return BiLSTM_Model3

        elif hidden_num == 2:

            return BiLSTM_Model2

        else:

            print(
                "hidden number not in [4, 3, 2], over range models. Please check and try again."
            )

            return np.nan

    elif model_label == "BiGRU":

        if hidden_num == 4:

            return BiGRU_Model

        elif hidden_num == 3:

            return BiGRU_Model3

        elif hidden_num == 2:

            return BiGRU_Model2

        else:

            print(
                "hidden number not in [4, 3, 2], over range models. Please check and try again."
            )

            return np.nan

    elif model_label == "BiCNN":

        if hidden_num == 4:

            return Conv2D_model

        elif hidden_num == 3:

            return Conv2D_model3

        elif hidden_num == 2:

            return Conv2D_model2

        else:

            print(
                "hidden number not in [4, 3, 2], over range models. Please check and try again."
            )

            return np.nan

    else:

        print(
            "model_label not in [BiLSTM, BiGRU, Conv2D], over range models. Please check and try again."
        )

        return np.nan


# 1.3、main.py

# -*-coding: utf-8 -*-

import time

import math

import tensorflow as tf

from data import *

from models import *

import warnings

warnings.filterwarnings("ignore")

# LR decay


def help_lr_decay(opter):

    if opter == "Nadam":

        initial_lrate = 0.002

    else:

        initial_lrate = 0.001

    return initial_lrate


def step_decay(epoch):

    drop = 0.5

    epochs_drop = 50

    initial_lrate = help_lr_decay(opter)

    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))

    lrate = max(lrate, 1e-5)

    return lrate


# custom function

from scipy.stats import spearmanr


def get_spearman_rankcor(y_true, y_pred):

    return tf.py_function(
        spearmanr,
        [tf.cast(y_pred, tf.float32), tf.cast(y_true, tf.float32)],
        Tout=tf.float32,
    )


def spearman(y_true, y_pred):

    import pandas as pd

    y_true = y_true.reshape(y_true.shape[0])

    y_pred = y_pred.reshape(y_true.shape[0])

    sp = pd.Series(y_pred).corr(pd.Series(y_true), method="spearman")

    return sp


# 均方差


def get_mse(records_real, records_predict):
    """

    均方误差 估计值与真值 偏差

    """

    if len(records_real) == len(records_predict):

        return sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len(
            records_real
        )

    else:

        return None


def model_fit(
    x_train,
    y_train,
    x_val,
    y_val,
    seq_len,
    model_label,
    hidden_num,
    special,
    params,
    epochs,
    early_stopping_patience,
    save_model_path,
):

    # intial model

    model = selecting_deeplearning(model_label, hidden_num, special)(params, seq_len)

    # compile

    model.compile(
        loss="mse", optimizer=params["optimizer"], metrics=["mse", get_spearman_rankcor]
    )

    # checkpoint

    from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping

    checkpoint_file = save_model_path

    checkpoint = ModelCheckpoint(
        filepath=checkpoint_file,
        monitor="val_get_spearman_rankcor",
        verbose=1,
        save_best_only=True,
        mode="max",
    )

    # LearningRateSchedule

    lrate = LearningRateScheduler(step_decay)

    # early stopping

    early_stopping = EarlyStopping(
        monitor="val_get_spearman_rankcor",
        patience=early_stopping_patience,
        verbose=0,
        mode="max",
    )

    callbacks_list = [checkpoint, lrate, early_stopping]

    # fit

    try:

        model.fit(
            x_train,
            y_train,
            batch_size=params["batch_size"],
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks_list,
            shuffle=True,
            verbose=0,
        )

    except OSError as e:

        print(e)

        pass


# recode model result


def recorde_model_result(
    save_params_hyperopt_log,
    model_path,
    x_train,
    x_val,
    y_train,
    y_val,
    model_label,
    hidden_num,
    val_fold_n,
    count,
    run_time,
    params,
):

    # get best model path

    model_dir = "/".join(model_path.split("/")[:-1])

    model_path_list = walk(model_dir)

    model_path = best_5_epoches_model(model_path_list, 1)[-1]

    from keras.models import load_model

    try:

        model = load_model(
            model_path, custom_objects={"get_spearman_rankcor": get_spearman_rankcor}
        )

        # 评估模型

        y_train_pred = model.predict(x_train)

        y_test_pred = model.predict(x_val)

        ## spearman

        train_spearman = spearman(y_train, y_train_pred)

        test_spearman = spearman(y_val, y_test_pred)

        ## mse

        train_loss_mse = get_mse(y_train, y_train_pred)

        test_loss_mse = get_mse(y_val, y_test_pred)

        print("Test spearman:", test_spearman)

        print("run_time:", run_time, "Using:", params)

    except ValueError as e:

        train_spearman, test_spearman, train_loss_mse, test_loss_mse = (
            0,
            0,
            1000,
            1000,
        )

        print("count =", count, "Error: ", e)

        print("run_time:", run_time, "Using:", params)

    model_id = "%s_hidden%s-%s" % (model_label, hidden_num, val_fold_n)

    temp_info = "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (
        model_id,
        count,
        run_time,
        train_spearman,
        test_spearman,
        train_loss_mse,
        test_loss_mse,
        str(params),
    )

    if count == 1:

        cols = "model_id\tmodel_count\trun_time\ttrain_spearman\ttest_spearman\ttrain_loss_mse\ttest_loss_mse\tparam\n"

        with open(save_params_hyperopt_log, "a") as a:

            a.write(cols)

    else:

        pass

    with open(save_params_hyperopt_log, "a") as a:

        a.write(temp_info)


## 主函数


def main(
    val_fold_n,
    x_train,
    y_train,
    x_val,
    y_val,
    seq_len,
    model_label,
    hidden_num,
    params,
    epochs,
    early_stopping_patience,
    save_dir,
):

    mkdir(save_dir)

    sub_save_dir = save_dir + "/val_fold_n_%s" % (val_fold_n)

    mkdir(sub_save_dir)

    save_params_hyperopt_log = sub_save_dir + "/summary_val_fold%s_performance.log" % (
        val_fold_n
    )

    is_Exist_file(save_params_hyperopt_log)

    print(params)

    # 训练 10 次对于每一个 model parameter，以应对不同随机初始化参数

    for count in range(10):

        start = time.time()

        # training

        save_model_path = (
            sub_save_dir
            + "/%s_%s-val_fold_n%s-count_%s-best_weights-improvement-{epoch:03d}-train-{get_spearman_rankcor:.5f}-test-{val_get_spearman_rankcor:.5f}.hdf5"
            % (model_label, hidden_num, val_fold_n, count)
        )

        print("\n%s model_fit ... " % (count))

        model_fit(
            x_train,
            y_train,
            x_val,
            y_val,
            seq_len,
            model_label,
            hidden_num,
            params,
            epochs,
            early_stopping_patience,
            save_model_path,
        )

        end = time.time()

        run_time = end - start

        # recording

        print("%s recording ... " % (count))

        recorde_model_result(
            save_params_hyperopt_log,
            save_model_path,
            x_train,
            x_val,
            y_train,
            y_val,
            model_label,
            hidden_num,
            val_fold_n,
            count,
            run_time,
            params,
        )


if __name__ == "__main__":

    import sys

    (
        main_path,
        read_fold_data_path,
        KFold_n,
        val_fold_n,
        seq_len,
        ycol,
        save_dir,
        model_label,
        hidden_num,
        early_stopping_patience,
        epochs,
    ) = sys.argv[1:]

    os.chdir(main_path)

    print("sys.executable:", sys.executable)

    print("sys.prefix:", sys.prefix)

    print("\n")

    # execute: train & test

    KFold_n = eval(KFold_n)

    val_fold_n = eval(val_fold_n)

    seq_len = int(seq_len)  # 28

    hidden_num = int(hidden_num)

    early_stopping_patience = int(early_stopping_patience)

    epochs = int(epochs)

    # get layer label

    if model_label == "BiCNN":

        layer_label = "2D"

    else:

        layer_label = "1D"

    # get model data

    x_train, y_train, x_val, y_val = main_model_data(
        read_fold_data_path, KFold_n, val_fold_n, seq_len, ycol, layer_label
    )

    print("Sequence features -- Finished")

    # 模型训练

    # 参数设置

    # get model params

    params = selection_model_parameter_tuple(model_label, hidden_num)

    opter = params["optimizer"]

    main(
        val_fold_n,
        x_train,
        y_train,
        x_val,
        y_val,
        seq_len,
        model_label,
        hidden_num,
        params,
        epochs,
        early_stopping_patience,
        save_dir,
    )


# AIdit_Cas9_OFF 模型训练

# 2.1、data.py

# -*-coding: utf-8 -*-

import os

import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings("ignore")


def is_Exist_file(path):

    import os

    if os.path.exists(path):

        os.remove(path)


def mkdir(path):

    import os

    path = path.strip()

    path = path.rstrip("\\")

    isExists = os.path.exists(path)

    if not isExists:

        os.makedirs(path)

    else:

        pass


def walk(path):

    import os

    input_path_list = []

    if not os.path.exists(path):

        return -1

    for root, dirs, names in os.walk(path):

        for filename in names:

            input_path = os.path.join(root, filename)

            input_path_list.append(input_path)

    return input_path_list


# 计算 a pair of sequences 之间的 mismatch 的个数


def compute_mismatch_number(seq1, seq2):

    mismatch_num = 0

    for index, nucle1 in enumerate(seq1):

        nucle2 = seq2[index]

        if nucle1 != nucle2:

            mismatch_num += 1

        else:

            pass

    return mismatch_num


# gRNASeq & offSeq 比对


def alignment_on_off_Sequence(gRNASeq, offSeq):

    gRNASeq = gRNASeq.upper()

    offSeq = offSeq.upper()

    align = []

    for index, nucle0 in enumerate(gRNASeq):

        nucle1 = offSeq[index]

        if nucle1 != "-":

            align.append(nucle0 + nucle1)

        else:

            align.append(nucle0 + nucle0)

    return "-".join(align)


# alignment: on-off deletion sequence


def alignment_on_off_deletion_sequence(new_offSeq_Target):

    on_off_deltSeq = ""

    for nucle in new_offSeq_Target:

        if nucle != "-":

            on_off_deltSeq = on_off_deltSeq + "."

        else:

            on_off_deltSeq = on_off_deltSeq + "-"

    return on_off_deltSeq


# 比对确定 off-target insertion sequence


def alignment_on_off_insertion_sequence(offSeq_Target):

    import re

    inser_nucles = re.findall("[acgt]", offSeq_Target)

    inser_nucles = list(set(inser_nucles))

    ##

    inserSeq = ""

    for index, nucle in enumerate(offSeq_Target):

        if nucle not in inser_nucles:

            inserSeq = inserSeq + "."

        else:

            inserSeq = inserSeq + nucle.upper()

    return inserSeq


# off-target mismatch/insertion/deletion modeling data


def main_off_target_Modeling_data(off_data, mut_type="mismatch"):

    off_data["gRNASeq"] = off_data["gRNASeq_63bp"].apply(lambda x: x[20:43])

    off_data["PAM-NN"] = off_data["offSeq_63bp"].apply(lambda x: x[41:43])

    off_data["on_off_alignSeq"] = off_data.apply(
        lambda row: alignment_on_off_Sequence(
            row["gRNASeq"], row["offSeq_63bp"][20:43]
        ),
        axis=1,
    )

    if mut_type == "mismatch":

        # compute mismatch number

        off_data["up_mismatch_num"] = off_data.apply(
            lambda row: compute_mismatch_number(
                row["gRNASeq_63bp"][:20], row["offSeq_63bp"][:20]
            ),
            axis=1,
        )

        off_data["core_mismatch_num"] = off_data.apply(
            lambda row: compute_mismatch_number(
                row["gRNASeq_63bp"][20:43], row["offSeq_63bp"][20:43]
            ),
            axis=1,
        )

        off_data["down_mismatch_num"] = off_data.apply(
            lambda row: compute_mismatch_number(
                row["gRNASeq_63bp"][-20:], row["offSeq_63bp"][-20:]
            ),
            axis=1,
        )

        cols = [
            "sgRNA_name",
            "new_mutation",
            "RW_off-target_eff",
            "BW_off-target_eff",
            "gRNASeq_63bp",
            "offSeq_63bp",
            "gRNASeq",
            "PAM-NN",
            "on_off_alignSeq",
            "up_mismatch_num",
            "core_mismatch_num",
            "down_mismatch_num",
            "on_pred",
            "off_pred",
        ]

    elif mut_type == "deletion":

        off_data["on_off_deltSeq"] = off_data["offSeq_63bp"].apply(
            lambda x: alignment_on_off_deletion_sequence(x[20:43])
        )

        cols = [
            "sgRNA_name",
            "new_mutation",
            "RW_off-target_eff",
            "BW_off-target_eff",
            "offSeq_63bp",
            "offSeq_28bp",
            "gRNASeq",
            "PAM-NN",
            "on_off_alignSeq",
            "on_off_deltSeq",
        ]

    elif mut_type == "insertion":

        off_data["on_off_inserSeq"] = off_data["offSeq_63bp"].apply(
            lambda x: alignment_on_off_insertion_sequence(x[20:43])
        )

        cols = [
            "sgRNA_name",
            "new_mutation",
            "RW_off-target_eff",
            "BW_off-target_eff",
            "offSeq_63bp",
            "offSeq_28bp",
            "gRNASeq",
            "PAM-NN",
            "on_off_alignSeq",
            "on_off_inserSeq",
        ]

    else:

        print(
            "Mutation type not in ['mismatch', 'insertion', 'deletion']. Please check and try again."
        )

        cols = [
            "sgRNA_name",
            "new_mutation",
            "RW_off-target_eff",
            "BW_off-target_eff",
            "gRNASeq_63bp",
            "offSeq_63bp",
            "gRNASeq",
            "PAM-NN",
            "on_off_alignSeq",
            "on_pred",
            "off_pred",
        ]

    data = off_data[cols]

    return data


# ********************* Feature one-hot Encoding ***********************

# 1、序列特征输入： 序列特征

# 生成 Seequence 数据


def find_all(sub, s):

    index = s.find(sub)

    feat_one = np.zeros(len(s))

    while index != -1:

        feat_one[index] = 1

        index = s.find(sub, index + 1)

    return feat_one


# 获取单样本序列数据


def obtain_each_seq_data(seq):

    A_array = find_all("A", seq)

    G_array = find_all("G", seq)

    C_array = find_all("C", seq)

    T_array = find_all("T", seq)

    one_sample = np.array([A_array, G_array, C_array, T_array])

    # print(one_sample.shape)

    return one_sample


#  获取序列数据

# 参数说明：

# data：输入的数据，要求含有 gRNA_28bp or gRNASeq_63bp 列名，该列为原始 DNA 序列

# 输出：特征数据 {'data': data}


def obtain_Sequence_data(data, seq_len=63, col="offSeq_63bp"):

    x_data = []

    for i, row in data.iterrows():

        seq = row[col]

        one_sample = obtain_each_seq_data(seq)

        one_sample_reshape = one_sample.T.reshape(seq_len * 4)

        # print(one_sample_reshape.shape)

        x_data.append(one_sample_reshape)

    # reshape

    x_data = np.array(x_data)

    x_data = x_data.astype("float32")

    return x_data


# 2、获得 PAM-NN 特征


def obtain_PAM_Feature(
    pam_nn, pam_feats=["GG", "AG", "GT", "GC", "GA", "TG", "CG", "other"]
):
    """

    pam_feats = ['GG', 'AG', 'GT', 'GC', 'GA', 'TG', 'CG', 'other']

    pam_nn = 'GG'

    pam_list = obtain_PAM_Feature(pam_nn, pam_feats)

    """

    pam_dict = {}

    for pam in pam_feats:

        pam_dict[pam] = 0

    if pam_nn in pam_dict:

        pam_dict[pam_nn] = 1

    else:

        pam_dict["other"] = 1

    # print(pam_dict)

    pam_list = []

    for pam in pam_feats:

        pam_list.append(pam_dict[pam])

    return pam_list


# 获得 PAM-NN 特征


def main_pam_data(data, pam_feats=["GG", "AG", "GT", "GC", "GA", "TG", "CG", "other"]):

    pam_data = []

    for index, row in data.iterrows():

        pam_nn = row["PAM-NN"]

        pam_list = obtain_PAM_Feature(pam_nn, pam_feats)

        pam_data.append(pam_list)

    ## pam data

    pam_data = np.array(pam_data)

    pam_data = pam_data.astype("float32")

    return pam_data


# 3、分解到每一个位置的 on-off mismatch feature

# 1、on-off alignment for position-substitution


def helper_each_position_alignSeq(one_pos_alignSeq):

    align_order = [
        "AC",
        "AG",
        "AT",
        "CA",
        "CG",
        "CT",
        "GA",
        "GC",
        "GT",
        "TA",
        "TC",
        "TG",
    ]

    align_list = []

    for one_align in align_order:

        if one_align == one_pos_alignSeq:

            align_list.append(1)

        else:

            align_list.append(0)

    return align_list


# one mismatch alignment sequence


def helper_one_alignSeq(alignSeq):
    """

    alignSeq = 'TT-TG-GC-CA-AG-GC-CA-AC-CC-AA-AA-CC-CC-CC-CC-AA-TT-GG-AA-AA-GG-GG-GG'

    all_align1 = helper_one_alignSeq(alignSeq)

    print(all_align1)

    """

    alignSeq_list = alignSeq.split("-")

    all_align_list = []

    for alignSeq in alignSeq_list:

        align_list = helper_each_position_alignSeq(alignSeq)

        all_align_list.append(align_list)

    all_align = np.array(all_align_list).T

    return all_align


# 获得 mismatch alignment feature


def main_mismatch_alignment_features(data):

    align_data = []

    for index, row in data.iterrows():

        alignSeq = row["on_off_alignSeq"]

        all_align = helper_one_alignSeq(alignSeq)

        all_align = all_align.T.reshape(all_align.shape[0] * all_align.shape[1])

        align_data.append(all_align)

    # align data

    align_data = np.array(align_data)

    align_data = align_data.astype("float32")

    return align_data


# 2、on-off alignment for only position


def helper_one_alignSeq_with_only_position(alignSeq):
    """

    alignSeq = 'TT-TG-GC-CA-AG-GC-CA-AC-CC-AA-AA-CC-CC-CC-CC-AA-TT-GG-AA-AA-GG-GG-GG'

    all_align_list2 = helper_one_alignSeq_with_only_position(alignSeq)

    print(all_align_list2)

    """

    alignSeq_list = alignSeq.split("-")

    all_align_list = []

    for alignSeq in alignSeq_list:

        if alignSeq[0] == alignSeq[1]:

            all_align_list.append(0)

        else:

            all_align_list.append(1)

    return all_align_list


# 获得 mismatch alignment feature with only position


def main_mismatch_alignment_features_with_only_position(data):

    align_data = []

    for index, row in data.iterrows():

        alignSeq = row["on_off_alignSeq"]

        all_align_list = helper_one_alignSeq_with_only_position(alignSeq)

        align_data.append(all_align_list)

    # align data

    align_data = np.array(align_data)

    align_data = align_data.astype("float32")

    return align_data


# 4、获得 on-off deletion position distribution


def helper_on_off_deletion_position(on_off_deltSeq):

    deltSeq_list = []

    for m in on_off_deltSeq:

        if m == ".":

            deltSeq_list.append(0)

        else:

            deltSeq_list.append(1)

    return deltSeq_list


# 获得 on-off deletion position feature


def main_on_off_deletion_position(data):

    delt_data = []

    for index, row in data.iterrows():

        on_off_deltSeq = row["on_off_deltSeq"]

        deltSeq_list = helper_on_off_deletion_position(on_off_deltSeq)

        delt_data.append(deltSeq_list)

    # delt data

    delt_data = np.array(delt_data)

    delt_data = delt_data.astype("float32")

    return delt_data


# 5、获得 on-off insertion position-nucleotide type


def help_on_off_insertion(on_off_inserSeq):

    ref_dict = {
        ".": [0, 0, 0, 0],
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
    }

    inserSeq_list = []

    for m in on_off_inserSeq[1:]:

        inserSeq_list.append(ref_dict[m])

    return inserSeq_list


# 获得 on-off insertion feature


def main_on_off_insertion_feature(data):

    inser_data = []

    for index, row in data.iterrows():

        on_off_inserSeq = row["on_off_inserSeq"]

        inserSeq_list = help_on_off_insertion(on_off_inserSeq)

        inserSeq = np.array(inserSeq_list)

        inserSeq = inserSeq.reshape(4 * (len(on_off_inserSeq) - 1))

        inser_data.append(inserSeq)

    # inser data

    inser_data = np.array(inser_data)

    inser_data = inser_data.astype("float32")

    return inser_data


# 仅考虑 insertion position


def help_on_off_insertion_position(on_off_inserSeq):

    inserSeq_pos = []

    for m in on_off_inserSeq[1:]:

        if m == ".":

            inserSeq_pos.append(0)

        else:

            inserSeq_pos.append(1)

    return inserSeq_pos


# 获得 on-off insertion feature position


def main_on_off_insertion_feature_woth_only_position(data):

    inser_data = []

    for index, row in data.iterrows():

        on_off_inserSeq = row["on_off_inserSeq"]

        inserSeq_pos = help_on_off_insertion_position(on_off_inserSeq)

        inser_data.append(inserSeq_pos)

    # inser data

    inser_data = np.array(inser_data)

    inser_data = inser_data.astype("float32")

    return inser_data


# mismatch

# 得到 off-target mismatch Feature Engineering

# must have: offSeq_63bp/offSeq_28bp, gRNASeq

# selective: PAM-NN,  on_off_alignSeq

# nparray_concat_to_one


def array_concat_to_one(collect_feat_data_dict):

    data = pd.DataFrame()

    for feat_label, array in collect_feat_data_dict.items():

        df_array = pd.DataFrame(array)

        cols_n = df_array.shape[1]

        cols = [feat_label + "_%s" % (i + 1) for i in range(cols_n)]

        df_array.columns = cols

        data = pd.concat([data, df_array], axis=1)

    return data


# deletion -- UPDATE

# 得到 off-target deletion Feature Engineering

# must have: offSeq_63bp/offSeq_28bp, gRNASeq

# selective: PAM-NN,  on_off_alignSeq, on_off_deltSeq

# feat_label 表示

# '+P': 'PAM-NN';

# '+M': 'on_off_alignSeq';

# '+Mp': 'on_off_alignSeq' with only position;

# '+D': 'on_off_deltSeq';

# '+P+M': 'PAM-NN + on_off_alignSeq';

# '+P+Mp': 'PAM-NN + on_off_alignSeq' with only position;

# '+P+D': 'PAM-NN + on_off_deltSeq';

# '+M+D': 'on_off_alignSeq + on_off_deltSeq';

# '+Mp+D': 'on_off_alignSeq + on_off_deltSeq' with only mismatch position;

# '+P+M+D': 'PAM-NN + on_off_alignSeq + on_off_deltSeq';

# '+P+Mp+D': 'PAM-NN + on_off_alignSeq + on_off_deltSeq' with mismath position;

# '+N': None.


def off_target_mismatch_feature_engineering(data, seq_len, fixed_feat, feat_label):

    # geting features: 'gRNASeq', 'PAM-NN', 'on_off_alignSeq'

    data = main_off_target_Modeling_data(data, mut_type="mismatch")

    if seq_len == 23:

        data["offSeq_23bp"] = data["offSeq_63bp"].apply(lambda x: x[20:43])

    else:

        pass

    pam_feats = ["GG", "AG", "GT", "GC", "GA", "TG", "CG", "other"]

    collect_feat_data_dict = {}

    # fixed feature list

    if fixed_feat == "seq_feat":

        x_data1 = obtain_Sequence_data(data, seq_len, col="offSeq_%sbp" % (seq_len))

        x_data2 = obtain_Sequence_data(data, seq_len=23, col="gRNASeq")

        collect_feat_data_dict["offSeq"] = x_data1

        collect_feat_data_dict["gRNASeq"] = x_data2

    elif fixed_feat == "mismatch_num":

        x_data3 = np.array(
            data[["up_mismatch_num", "core_mismatch_num", "down_mismatch_num"]]
        )

        collect_feat_data_dict["mismatch_num_region"] = x_data3

    elif fixed_feat == "pred_feat":

        pred_data = np.array(data[["on_pred", "off_pred"]])

        collect_feat_data_dict["pred_feat"] = pred_data

    elif fixed_feat == "seq_feat+mismatch_num":

        x_data1 = obtain_Sequence_data(data, seq_len, col="offSeq_%sbp" % (seq_len))

        x_data2 = obtain_Sequence_data(data, seq_len=23, col="gRNASeq")

        x_data3 = np.array(
            data[["up_mismatch_num", "core_mismatch_num", "down_mismatch_num"]]
        )

        collect_feat_data_dict["offSeq"] = x_data1

        collect_feat_data_dict["gRNASeq"] = x_data2

        collect_feat_data_dict["mismatch_num_region"] = x_data3

    elif fixed_feat == "pred_feat+mismatch_num":

        pred_data = np.array(data[["on_pred", "off_pred"]])

        x_data3 = np.array(
            data[["up_mismatch_num", "core_mismatch_num", "down_mismatch_num"]]
        )

        collect_feat_data_dict["pred_feat"] = pred_data

        collect_feat_data_dict["mismatch_num_region"] = x_data3

    elif fixed_feat == "all":

        x_data1 = obtain_Sequence_data(data, seq_len, col="offSeq_%sbp" % (seq_len))

        x_data2 = obtain_Sequence_data(data, seq_len=23, col="gRNASeq")

        x_data3 = np.array(
            data[["up_mismatch_num", "core_mismatch_num", "down_mismatch_num"]]
        )

        pred_data = np.array(data[["on_pred", "off_pred"]])

        collect_feat_data_dict["offSeq"] = x_data1

        collect_feat_data_dict["gRNASeq"] = x_data2

        collect_feat_data_dict["mismatch_num_region"] = x_data3

        collect_feat_data_dict["pred_feat"] = pred_data

    else:  # None

        pass

    # Additional features

    if feat_label == "+P":

        pam_data = main_pam_data(data, pam_feats)

        collect_feat_data_dict["+P"] = pam_data

    elif feat_label == "+M":

        align_data = main_mismatch_alignment_features(data)

        collect_feat_data_dict["+M"] = align_data

    elif feat_label == "+Mp":  # consider mismatch position

        align_data = main_mismatch_alignment_features_with_only_position(data)

        collect_feat_data_dict["+Mp"] = align_data

    elif feat_label == "+P+M":

        pam_data = main_pam_data(data, pam_feats)

        align_data = main_mismatch_alignment_features(data)

        collect_feat_data_dict["+P"] = pam_data

        collect_feat_data_dict["+M"] = align_data

    elif feat_label == "+P+Mp":

        pam_data = main_pam_data(data, pam_feats)

        align_data = main_mismatch_alignment_features_with_only_position(data)

        collect_feat_data_dict["+P"] = pam_data

        collect_feat_data_dict["+Mp"] = align_data

    else:  # None

        pass

    # feature concating

    xdata = array_concat_to_one(collect_feat_data_dict)

    return xdata


# 2.2、model.py

# -*-coding: utf-8 -*-

import os

import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings("ignore")


# For Prediction & Evaluation


def evaluation_for_conventional_meachine_learning(
    model, x_train, y_train, x_val, y_val
):

    from sklearn.metrics import mean_squared_error

    # 0prediction

    y_train_pred = model.predict(x_train)

    y_val_pred = model.predict(x_val)

    # evaluate

    # spearman

    train_spearman = pd.Series(y_train_pred).corr(pd.Series(y_train), method="spearman")

    val_spearman = pd.Series(y_val_pred).corr(pd.Series(y_val), method="spearman")

    # pearson

    train_pccs = pd.Series(y_train_pred).corr(pd.Series(y_train), method="pearson")

    val_pccs = pd.Series(y_val_pred).corr(pd.Series(y_val), method="pearson")

    # MSE

    train_mse = mean_squared_error(y_train, y_train_pred)

    val_mse = mean_squared_error(y_val, y_val_pred)

    print("train_spearman:", train_spearman, "val_spearman", val_spearman)

    return (train_spearman, val_spearman, train_pccs, val_pccs, train_mse, val_mse)


# For Elastic


def Elastic_for_off_target_Modeling(x_train, y_train, x_val, y_val, params):

    from sklearn.linear_model import ElasticNet

    # create model & fit the model

    model = ElasticNet(**params)

    model.fit(x_train, y_train)

    # prediction and evaluation

    train_spearman, val_spearman, train_pccs, val_pccs, train_mse, val_mse = (
        evaluation_for_conventional_meachine_learning(
            model, x_train, y_train, x_val, y_val
        )
    )

    return (
        train_spearman,
        val_spearman,
        train_pccs,
        val_pccs,
        train_mse,
        val_mse,
        model,
    )


# For Ridge


def Ridge_for_off_target_Modeling(x_train, y_train, x_val, y_val, params):

    from sklearn.linear_model import Ridge

    # create model & fit the model

    model = Ridge(**params)

    model.fit(x_train, y_train)

    # prediction and evaluation

    train_spearman, val_spearman, train_pccs, val_pccs, train_mse, val_mse = (
        evaluation_for_conventional_meachine_learning(
            model, x_train, y_train, x_val, y_val
        )
    )

    return (
        train_spearman,
        val_spearman,
        train_pccs,
        val_pccs,
        train_mse,
        val_mse,
        model,
    )


# For Lasso


def Lasso_for_off_target_Modeling(x_train, y_train, x_val, y_val, params):

    from sklearn.linear_model import Lasso

    # create model & fit the model

    model = Lasso(**params)

    model.fit(x_train, y_train)

    # prediction and evaluation

    train_spearman, val_spearman, train_pccs, val_pccs, train_mse, val_mse = (
        evaluation_for_conventional_meachine_learning(
            model, x_train, y_train, x_val, y_val
        )
    )

    return (
        train_spearman,
        val_spearman,
        train_pccs,
        val_pccs,
        train_mse,
        val_mse,
        model,
    )


# For XGBoost


def XGBoost_for_off_target_Modeling(x_train, y_train, x_val, y_val, params):

    from xgboost import XGBRegressor

    # create model & fit the model

    model = XGBRegressor(**params)

    model.fit(x_train, y_train)

    # prediction and evaluation

    train_spearman, val_spearman, train_pccs, val_pccs, train_mse, val_mse = (
        evaluation_for_conventional_meachine_learning(
            model, x_train, y_train, x_val, y_val
        )
    )

    return (
        train_spearman,
        val_spearman,
        train_pccs,
        val_pccs,
        train_mse,
        val_mse,
        model,
    )


# For MLP


def MLP_for_off_target_Modeling(x_train, y_train, x_val, y_val, params):

    from sklearn.neural_network import MLPRegressor

    # create model & fit the model

    hidden_layer_sizes = (
        params["hidden_layer_sizes_1"],
        params["hidden_layer_sizes_2"],
        params["hidden_layer_sizes_3"],
        params["hidden_layer_sizes_4"],
        params["hidden_layer_sizes_5"],
    )

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=params["alpha"],
        max_iter=params["max_iter"],
        random_state=2020,
        shuffle=True,
        verbose=False,
        activation="relu",
        solver="adam",
        learning_rate="invscaling",
    )

    model.fit(x_train, y_train)

    # prediction and evaluation

    train_spearman, val_spearman, train_pccs, val_pccs, train_mse, val_mse = (
        evaluation_for_conventional_meachine_learning(
            model, x_train, y_train, x_val, y_val
        )
    )

    return (
        train_spearman,
        val_spearman,
        train_pccs,
        val_pccs,
        train_mse,
        val_mse,
        model,
    )


# 2.3、main.py

# -*-coding: utf-8 -*-

from data import *

from models import *

import warnings

warnings.filterwarnings("ignore")


# recorder


def recorder(
    count,
    cell_line,
    reads_cutoff,
    model_label,
    fixed_feat,
    feat_label,
    run_time,
    params,
    save_model_path,
    save_results_path,
    train_spearman,
    val_spearman,
    train_pccs,
    val_pccs,
    train_mse,
    val_mse,
    model,
):

    import joblib

    joblib.dump(model, save_model_path)  # 也可以使用文件对象

    if not os.path.exists(save_results_path):

        cols = "count\tcell_line\treads_cutoff\tmodel_label\tfixed_feat\tfeat_label\trun_time\ttrain_mse\tval_mse\ttrain_pearson\tval_pearson\ttrain_spearman\tval_spearman\tparams\n"

        with open(save_results_path, "a") as a:

            a.write(cols)

    else:

        pass

    temp_info = "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (
        count,
        cell_line,
        reads_cutoff,
        model_label,
        fixed_feat,
        feat_label,
        run_time,
        train_mse,
        val_mse,
        train_pccs,
        val_pccs,
        train_spearman,
        val_spearman,
        str(params),
    )

    with open(save_results_path, "a") as a:

        a.write(temp_info)


# training


def main_modeling(
    x_train,
    y_train,
    x_val,
    y_val,
    save_model_path,
    save_results_path,
    count,
    cell_line,
    reads_cutoff,
    model_label,
    fixed_feat,
    feat_label,
    params,
):

    import time

    start = time.time()

    # training

    params = eval(params)

    if model_label == "Lasso":

        results = Lasso_for_off_target_Modeling(x_train, y_train, x_val, y_val, params)

    elif model_label == "Ridge":

        results = Ridge_for_off_target_Modeling(x_train, y_train, x_val, y_val, params)

    elif model_label == "Elastic":

        results = Elastic_for_off_target_Modeling(
            x_train, y_train, x_val, y_val, params
        )

    elif model_label == "XGBoost":

        results = XGBoost_for_off_target_Modeling(
            x_train, y_train, x_val, y_val, params
        )

    elif model_label == "MLP":

        results = MLP_for_off_target_Modeling(x_train, y_train, x_val, y_val, params)

    else:

        results = ""

    train_spearman, val_spearman, train_pccs, val_pccs, train_mse, val_mse, model = (
        results
    )

    end = time.time()

    run_time = round(end - start, 3)

    # recorder

    params = str(params)

    recorder(
        count,
        cell_line,
        reads_cutoff,
        model_label,
        fixed_feat,
        feat_label,
        run_time,
        params,
        save_model_path,
        save_results_path,
        train_spearman,
        val_spearman,
        train_pccs,
        val_pccs,
        train_mse,
        val_mse,
        model,
    )


if __name__ == "__main__":

    import sys

    (
        main_path,
        read_train_path,
        read_val_path,
        read_params_path,
        y_col,
        seq_len,
        count,
        cell_line,
        reads_cutoff,
        model_label,
        fixed_feat,
        feat_label,
        save_dir,
    ) = sys.argv[1:]

    os.chdir(main_path)

    # Get Train & Validation Data

    seq_len = eval(seq_len)

    train_data = pd.read_csv(read_train_path, sep="\t")

    val_data = pd.read_csv(read_val_path, sep="\t")

    x_train = off_target_mismatch_feature_engineering(
        train_data, seq_len, fixed_feat, feat_label
    )

    x_val = off_target_mismatch_feature_engineering(
        val_data, seq_len, fixed_feat, feat_label
    )

    x_train, y_train = np.array(x_train), np.array(train_data[y_col])

    x_val, y_val = np.array(x_val), np.array(val_data[y_col])

    # training

    reads_cutoff = eval(reads_cutoff)

    model_params = pd.read_csv(read_params_path, sep="\t")

    one_params = model_params.loc[
        (model_params["cell_line"] == cell_line)
        & (model_params["reads_cutoff"] == reads_cutoff)
        & (model_params["model"] == model_label)
        & (model_params["fixed_feat"] == fixed_feat)
        & (model_params["feat_label"] == feat_label),
        :,
    ]

    index = one_params.index.tolist()[0]

    params = str(one_params.loc[index, "params"])

    print("\nCell line: %s; \nparameter: %s" % (cell_line, params))

    mkdir(save_dir)

    save_model_path = save_dir + "/%s-%s_%s-%s%s_count-%s.model" % (
        cell_line,
        reads_cutoff,
        model_label,
        fixed_feat,
        feat_label,
        count,
    )

    save_results_path = save_dir + "/summary_for_off-target_modeling.log"

    main_modeling(
        x_train,
        y_train,
        x_val,
        y_val,
        save_model_path,
        save_results_path,
        count,
        cell_line,
        reads_cutoff,
        model_label,
        fixed_feat,
        feat_label,
        params,
    )


# AIdit_Cas9_DSB 模型训练

# 3.1、XGBoost for each category

# -*-coding: utf-8 -*-

import os

import time

import numpy as np

import pandas as pd

from DSB_Repair_Feature_and_Categories import *

import warnings

warnings.filterwarnings("ignore")


# 4.1 Some Evaluation Functions

# Model 3: XGBoost


def XGBoost_fit(Xtrain, ytrain, params):

    from xgboost import XGBRegressor

    # create model & fit the model

    model = XGBRegressor(**params)

    model.fit(Xtrain, ytrain)

    return model


# 0prediction & evaluation


def Evaluation(model, Xdata, ydata):

    # 0prediction

    ypred = model.predict(Xdata)

    # evaluation

    eval_pearson = pd.Series(ypred).corr(pd.Series(ydata), method="pearson")

    eval_spearman = pd.Series(ypred).corr(pd.Series(ydata), method="spearman")

    return eval_pearson, eval_spearman


def main_XGBoost(Xtrain, ytrain, Xval, yval, model_params, save_model_path):

    from sklearn.externals import joblib

    # training

    model = XGBoost_fit(Xtrain, ytrain, model_params)

    joblib.dump(model, save_model_path)

    # 0prediction & evaluation

    train_pearson, train_spearman = Evaluation(model, Xtrain, ytrain)

    val_pearson, val_spearman = Evaluation(model, Xval, yval)

    print("train_spearman:", train_spearman, "val_spearman", val_spearman)

    return (train_pearson, train_spearman, val_pearson, val_spearman)


def Obtain_model_params(Cell_Line, seq_len):

    #  model parameter

    if (Cell_Line == "K562") & (seq_len == 28):

        model_param = {
            "n_estimators": 2369,
            "nthread": 25,
            "learning_rate": 0.0578,
            "max_depth": 9,
            "max_leaf_nodes": 185,
            "colsample_bytree": 0.784,
            "subsample": 0.967,
            "reg_alpha": 2.499,
            "reg_lambda": 34.591,
        }

    elif (Cell_Line == "K562") & (seq_len == 63):

        model_param = {
            "n_estimators": 2249,
            "nthread": 25,
            "learning_rate": 0.0439,
            "max_depth": 9,
            "max_leaf_nodes": 164,
            "colsample_bytree": 0.819,
            "subsample": 0.999,
            "reg_alpha": 1.273,
            "reg_lambda": 33.017,
        }

    elif (Cell_Line == "Jurkat") & (seq_len == 28):

        model_param = {
            "n_estimators": 2794,
            "nthread": 25,
            "learning_rate": 0.0713,
            "max_depth": 8,
            "max_leaf_nodes": 255,
            "colsample_bytree": 0.723,
            "subsample": 0.949,
            "reg_alpha": 4.177,
            "reg_lambda": 32.991,
        }

    elif (Cell_Line == "Jurkat") & (seq_len == 63):

        model_param = {
            "n_estimators": 2600,
            "nthread": 25,
            "learning_rate": 0.062,
            "max_depth": 8,
            "max_leaf_nodes": 37,
            "colsample_bytree": 0.797,
            "subsample": 0.954,
            "reg_alpha": 1.257,
            "reg_lambda": 8.984,
        }

    else:

        model_param = {}

        print(
            "Input Error: Cell Line expected in ['K562', 'Jurkat'] and seq_len = 28 0r 63, "
            "but Cell_Line=%s, seq_len=%s" % (Cell_Line, seq_len)
        )

    return model_param


# Get train & test data


def Obtain_predicting_feature(data, seq_bp=28, max_len=30):

    # 1. to get sequence feature

    seq_data = obtain_single_sequence_one_hot_feature_2nd(data, seq_bp)

    # 2. to get MH feature

    edit_sites = [34, 35, 36, 37, 38, 39, 40]

    MH_data = main_MH_Feature_2nd(data, edit_sites, max_len)

    # 3. merge data

    Xdata = pd.merge(seq_data, MH_data, how="inner", on=["sgRNA_name", "gRNASeq_85bp"])

    del Xdata["sgRNA_name"]

    del Xdata["gRNASeq_85bp"]

    return np.array(Xdata)


def Obtain_predicting_categories_1(data, y_col):

    # label_list = list(int_data['new category'].unique())

    ydata = np.array(data[y_col])

    return ydata


def main(Cell_Line, train_data, val_data, y_col, save_dir, seq_bp=63, max_len=30):

    # save path

    save_model_dir = save_dir + "/%s-%sbp" % (Cell_Line, seq_bp)

    mkdir(save_model_dir)

    save_model_path = save_model_dir + "/XGB_%s-%sbp_%s.model" % (
        Cell_Line,
        seq_bp,
        y_col,
    )

    save_summary_path = save_dir + "/Summary_%s-DSB-Modeling.log" % (Cell_Line)

    # Get train & validation data

    Xtrain = Obtain_predicting_feature(train_data, seq_bp, max_len)

    ytrain = Obtain_predicting_categories_1(train_data, y_col)

    Xval = Obtain_predicting_feature(val_data, seq_bp, max_len)

    yval = Obtain_predicting_categories_1(val_data, y_col)

    print("----------------------")

    print("Xtrain.shape:", Xtrain.shape, "; ytrain.shape:", ytrain.shape)

    print("Xval.shape:", Xval.shape, "; yval.shape:", yval.shape)

    Data_Count = Xtrain.shape[0]

    # Training

    start = time.time()

    model_param = Obtain_model_params(Cell_Line, seq_bp)

    train_pearson, train_spearman, val_pearson, val_spearman = main_XGBoost(
        Xtrain, ytrain, Xval, yval, model_param, save_model_path
    )

    end = time.time()

    using_time = end - start

    # Write

    localtime = time.asctime(time.localtime(time.time()))

    if os.path.exists(save_summary_path):

        pass

    else:

        with open(save_summary_path, "a") as a:

            col_info = "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (
                "Cell Line",
                "Data Count",
                "Sequence Length",
                "y_col",
                "Train Pearson",
                "Train Spearman",
                "Val Pearson",
                "Val Spearman",
                "Model Parameter",
                "Using Time",
                "Writing Time",
            )

            a.write(col_info)

    with open(save_summary_path, "a") as a:

        # write

        text_info = "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (
            Cell_Line,
            Data_Count,
            seq_bp,
            y_col,
            train_pearson,
            train_spearman,
            val_pearson,
            val_spearman,
            str(model_param),
            using_time,
            localtime,
        )

        a.write(text_info)

    print("Finish.")


# 3.2、DSB repair engineered feature

# -*-coding: utf-8 -*-

from DSB_Repair_Feature_and_Categories import *

import warnings

warnings.filterwarnings("ignore")


def Obtain_predicting_feature_2nd(data, seq_bp=28, max_len=30):

    ## 1. to get sequence feature

    seq_data = obtain_single_sequence_one_hot_feature_2nd(data, seq_bp)

    ## 2. to get MH feature

    edit_sites = [34, 35, 36, 37, 38, 39, 40]

    MH_data = main_MH_Feature_2nd(data, edit_sites, max_len)

    return (seq_data, MH_data)


def Obtain_predicting_categories_1(data, y_col):

    # label_list = list(int_data['new category'].unique())

    ydata = np.array(data[y_col])

    return ydata


# 0prediction


def xgb_prediction(Xdata, model_path):

    from sklearn.externals import joblib

    model = joblib.load(model_path)  # 加载

    ypred = model.predict(Xdata)

    return ypred


# data.columns: ['sgRNA_name', 'gRNASeq_85bp'] at least


def main_xgb_prediction(data, int_data, model_path_pattern, seq_bp=63, max_len=30):

    # Get Xdata

    seq_data, MH_data = Obtain_predicting_feature_2nd(data, seq_bp, max_len)

    Xdata = pd.merge(seq_data, MH_data, how="inner", on=["sgRNA_name", "gRNASeq_85bp"])

    del Xdata["sgRNA_name"]

    del Xdata["gRNASeq_85bp"]

    Xdata = np.array(Xdata)

    print("----------------------")

    print("Xtrain.shape:", Xdata.shape)

    # Get Engineered Feature

    # model_path = "XGB_K562-63bp_%s.model"%("29:40D-12")

    eng_data = seq_data[["sgRNA_name", "gRNASeq_85bp"]]

    for model_label in int_data["new category"].unique():

        temp_model_path = model_path_pattern % (model_label)

        ypred = xgb_prediction(Xdata, temp_model_path)

        eng_data[model_label] = ypred

    return (seq_data, MH_data, eng_data)


# 3.3、DSB repair features and categories

# -*-coding: utf-8 -*-

import os

import numpy as np

import pandas as pd

import warnings

warnings.filterwarnings("ignore")


# 基础功能 1：删除文件和创建文件夹

# 检查文件是否存在，存在删除


def is_Exist_file(path):

    import os

    if os.path.exists(path):

        os.remove(path)


def mkdir(path):

    import os

    path = path.strip()

    path = path.rstrip("\\")

    isExists = os.path.exists(path)

    if not isExists:

        os.makedirs(path)


# 获取单位置核苷酸特征

# 基础功能 2：  Get sequence Feature

# get_dummies: feature one-hot encoding


def Get_Dummies(df, feature_list):

    df_dummies = pd.get_dummies(df[feature_list], columns=feature_list, prefix_sep="-")

    # 去除 df 中含有 df_dummies 的列

    for col in df_dummies.columns.tolist():

        if col in df.columns.tolist():

            del df[col]

    # concat

    df = pd.concat([df, df_dummies], axis=1)

    return df


def helper_single_feature_list(raw_data, seq_bp):

    raw_data["gRNAUp"] = raw_data["gRNASeq_85bp"].apply(lambda x: x[:20])

    raw_data["gRNATarget"] = raw_data["gRNASeq_85bp"].apply(lambda x: x[20:40])

    raw_data["PAM"] = raw_data["gRNASeq_85bp"].apply(lambda x: x[40:43])

    raw_data["gRNADown"] = raw_data["gRNASeq_85bp"].apply(lambda x: x[43:63])

    # 单位置核苷酸特征

    single_feature_list = []

    if seq_bp == 63:

        # Up

        for i in range(20):

            raw_data["S-U%s" % (i + 1)] = raw_data["gRNAUp"].apply(lambda x: x[i])

            single_feature_list.append("S-U%s" % (i + 1))

        # Target

        for i in range(20):

            raw_data["S-T%s" % (i + 1)] = raw_data["gRNATarget"].apply(lambda x: x[i])

            single_feature_list.append("S-T%s" % (i + 1))

        # PAM

        raw_data["S-PAM(N)"] = raw_data["PAM"].apply(lambda x: x[0])

        single_feature_list.append("S-PAM(N)")

        # Down

        for i in range(20):

            raw_data["S-D(-%s)" % (i + 1)] = raw_data["gRNADown"].apply(lambda x: x[i])

            single_feature_list.append("S-D(-%s)" % (i + 1))

    else:  # 28bp

        # Target

        for i in range(20):

            raw_data["S-T%s" % (i + 1)] = raw_data["gRNATarget"].apply(lambda x: x[i])

            single_feature_list.append("S-T%s" % (i + 1))

        # PAM

        raw_data["S-PAM(N)"] = raw_data["PAM"].apply(lambda x: x[0])

        single_feature_list.append("S-PAM(N)")

        # Down

        for i in range(5):

            raw_data["S-D(-%s)" % (i + 1)] = raw_data["gRNADown"].apply(lambda x: x[i])

            single_feature_list.append("S-D(-%s)" % (i + 1))

    del raw_data["gRNAUp"]

    del raw_data["gRNATarget"]

    del raw_data["PAM"]

    del raw_data["gRNADown"]

    return single_feature_list


def obtain_single_sequence_one_hot_feature_2nd(data, seq_bp):

    import time

    raw_data = data[["sgRNA_name", "gRNASeq_85bp"]]

    print("================================")

    print("Function: Obtain_Single_Sequence_One_Hot_Feature ... ...")

    s = time.time()

    single_feature_list = helper_single_feature_list(raw_data, seq_bp)

    raw_data = Get_Dummies(raw_data, single_feature_list)

    # check all one-hot features in raw_data.columns & complement

    import copy

    single_one_hot_feature_list = []

    nfeat_list = copy.deepcopy(raw_data.columns.tolist())

    for feat in single_feature_list:

        for nucle in ["A", "C", "G", "T"]:

            one_hot_feat = feat + "-" + nucle

            single_one_hot_feature_list.append(one_hot_feat)

            if one_hot_feat not in nfeat_list:  # 补充不完整的 one-hot 特征

                raw_data[one_hot_feat] = 0

        del raw_data[feat]  # 删除非 one-hot 过渡特征

    raw_data = raw_data[["sgRNA_name", "gRNASeq_85bp"] + single_one_hot_feature_list]

    e = time.time()

    print("Using Time: %s" % (e - s))

    return raw_data


# 获取微同源特征

# 基础功能 3：  Get MH Feature

# Deletion Classes


def deletion_classes(edit_sites, max_len=30):

    min_site = min(edit_sites)

    max_site = max(edit_sites)

    delt_classes = []

    for i in range(1, max_len):

        inf_site = min_site - i + 1

        for site in range(inf_site, max_site + 1):

            key = "%s:%sD-%s" % (site, site + i - 1, i)

            delt_classes.append(key)

    delt_classes.append("D%s+" % (max_len))

    return delt_classes


# get MH feature

# Get 1bp MH


def help_1bp_MH(delt_inf_site, delt_sup_site, gRNASeq):

    delt_nucle = gRNASeq[delt_inf_site]

    MH_nucle = gRNASeq[delt_sup_site + 1]

    if delt_nucle == MH_nucle:

        MH = 1

    else:

        MH = 0

    return MH


# Get 2bp MH


def help_2bp_MH(delt_inf_site, delt_sup_site, gRNASeq):

    delt_nucles = gRNASeq[delt_inf_site : (delt_inf_site + 2)]

    MH_nucles = gRNASeq[(delt_sup_site + 1) : (delt_sup_site + 3)]

    if delt_nucles == MH_nucles:

        MH = 1

    else:

        MH = 0

    return MH


# Get 3bp MH


def help_3bp_MH(delt_inf_site, delt_sup_site, gRNASeq):

    delt_nucles = gRNASeq[delt_inf_site : (delt_inf_site + 3)]

    MH_nucles = gRNASeq[(delt_sup_site + 1) : (delt_sup_site + 4)]

    if delt_nucles == MH_nucles:

        MH = 1

    else:

        MH = 0

    return MH


# Get 4bp MH


def help_4bp_MH(delt_inf_site, delt_sup_site, gRNASeq):

    delt_nucles = gRNASeq[delt_inf_site : (delt_inf_site + 4)]

    MH_nucles = gRNASeq[(delt_sup_site + 1) : (delt_sup_site + 5)]

    if delt_nucles == MH_nucles:

        MH = 1

    else:

        MH = 0

    return MH


# MH: 1bp, 2bp, 3bp

# get MH feature


def Get_MH_Feature(gRNASeq, delt_classes, max_len=30):

    MH_feat_dict = {}

    for one_class in delt_classes:

        if one_class != "D%s+" % (max_len):

            delt_len = int(one_class.split("-")[1])

            delt_p = one_class.split("-")[0]

            delt_inf_site = int(delt_p.split(":")[0]) - 1

            delt_sup_site = int(delt_p.split(":")[1][:-1]) - 1

            if delt_len == 1:

                MH = help_1bp_MH(delt_inf_site, delt_sup_site, gRNASeq)

                MH_feat_dict["%s_1bp" % (one_class)] = MH

            elif delt_len == 2:

                MH1 = help_1bp_MH(delt_inf_site, delt_sup_site, gRNASeq)

                MH2 = help_2bp_MH(delt_inf_site, delt_sup_site, gRNASeq)

                MH_feat_dict["%s_1bp" % (one_class)] = MH1

                MH_feat_dict["%s_2bp" % (one_class)] = MH2

            elif delt_len == 3:

                MH1 = help_1bp_MH(delt_inf_site, delt_sup_site, gRNASeq)

                MH2 = help_2bp_MH(delt_inf_site, delt_sup_site, gRNASeq)

                MH3 = help_3bp_MH(delt_inf_site, delt_sup_site, gRNASeq)

                MH_feat_dict["%s_1bp" % (one_class)] = MH1

                MH_feat_dict["%s_2bp" % (one_class)] = MH2

                MH_feat_dict["%s_3bp" % (one_class)] = MH3

            else:

                MH1 = help_1bp_MH(delt_inf_site, delt_sup_site, gRNASeq)

                MH2 = help_2bp_MH(delt_inf_site, delt_sup_site, gRNASeq)

                MH3 = help_3bp_MH(delt_inf_site, delt_sup_site, gRNASeq)

                MH4 = help_4bp_MH(delt_inf_site, delt_sup_site, gRNASeq)

                MH_feat_dict["%s_1bp" % (one_class)] = MH1

                MH_feat_dict["%s_2bp" % (one_class)] = MH2

                MH_feat_dict["%s_3bp" % (one_class)] = MH3

                MH_feat_dict["%s_4bp" % (one_class)] = MH4

        else:

            pass

    # sorting

    keys = list(MH_feat_dict.keys())

    keys.sort(reverse=False)

    MH_feats = [MH_feat_dict[key] for key in keys]

    return (MH_feats, keys)


# adjust two gRNASeq_85bp


def adjust_column_gRNASeq_85bp(gRNA_name, gRNASeq_85bp):

    if (gRNA_name != "AC026748.1_5_ENST00000624349_ATCAGCGCTGAGCCCATCAG") & (
        gRNA_name != "AC026748.1_4_ENST00000624349_AGCGCTGATGGGCTCAGCGC"
    ):

        return gRNASeq_85bp

    else:

        if (gRNA_name == "AC026748.1_5_ENST00000624349_ATCAGCGCTGAGCCCATCAG") & (
            gRNASeq_85bp is np.nan
        ):

            return "AGCTATAGGTCCAAGAGCCCATCAGCGCTGAGCCCATCAGCGCTGAGCCCATCAGGGGCTCGTGTCCTGGGCCTCCGGGTTTGTATTACCGCCATGCATT"

        elif (gRNA_name == "AC026748.1_4_ENST00000624349_AGCGCTGATGGGCTCAGCGC") & (
            gRNASeq_85bp is np.nan
        ):

            return "AGCTATAGGTCCAAGGGCTCAGCGCTGATGGGCTCAGCGCTGATGGGCTCAGCGCTGGGCTTGAGAGCAGGAGTGTGTGTTTGTATTACCGCCATGCATT"

        else:

            return gRNASeq_85bp


# 主函数: Get MH feature


def main_MH_Feature_2nd(data, edit_sites, max_len=30):

    df = data[["sgRNA_name", "gRNASeq_85bp"]]

    # adjust two gRNASeq_85bp

    df["gRNASeq_85bp"] = df.apply(
        lambda row: adjust_column_gRNASeq_85bp(row["sgRNA_name"], row["gRNASeq_85bp"]),
        axis=1,
    )

    # get mh features

    delt_classes = deletion_classes(edit_sites, max_len)

    df["MH_features"] = df["gRNASeq_85bp"].apply(
        lambda x: Get_MH_Feature(x, delt_classes, max_len)[0]
    )

    MH_data = pd.DataFrame(list(np.array(df["MH_features"])))

    # Get columns

    gRNASeq_85bp = "AGCTATAGGTCCAAGAGCCCATCAGCGCTGAGCCCATCAGCGCTGAGCCCATCAGGGGCTCGTGTCCTGGGCCTCCGGGTTTGTATTACCGCCATGCATT"

    cols = Get_MH_Feature(gRNASeq_85bp, delt_classes, max_len)[1]

    MH_data.columns = cols

    # concat

    del df["MH_features"]

    df = pd.concat([df[["sgRNA_name", "gRNASeq_85bp"]], MH_data], axis=1)

    return df


# Get train & test data


def Obtain_predicting_feature(data, seq_bp=28, max_len=30):

    # 1. to get sequence feature

    seq_data = obtain_single_sequence_one_hot_feature_2nd(data, seq_bp)

    # 2. to get MH feature

    edit_sites = [34, 35, 36, 37, 38, 39, 40]

    MH_data = main_MH_Feature_2nd(data, edit_sites, max_len)

    # 3. merge data

    Xdata = pd.merge(seq_data, MH_data, how="inner", on=["sgRNA_name", "gRNASeq_85bp"])

    del Xdata["sgRNA_name"]

    del Xdata["gRNASeq_85bp"]

    return np.array(Xdata)


def Obtain_predicting_categories(data, int_data):

    label_list = list(int_data["new category"].unique())

    ydata = np.array(data[label_list])

    return ydata


# 3.4、Modeling

# -*-coding: utf-8 -*-

from DSB_Repair_Engineered_Feature import *

import warnings

warnings.filterwarnings("ignore")


# 需要遍历的目录树的路径

# 路径和文件名连接构成完整路径


def walk(path):

    import os

    input_path_list = []

    if not os.path.exists(path):

        return -1

    for root, dirs, names in os.walk(path):

        for filename in names:

            input_path = os.path.join(root, filename)

            input_path_list.append(input_path)

    return input_path_list


# get best_checkpoint_path file


def get_best_checkpoint_path(path):

    file_list = walk(path)

    epoch_dict = {}

    for file in file_list:

        file_p = file.split("-")

        epoch = file_p[2]

        epoch_dict[epoch] = file

    epoch_max = max(list(epoch_dict.keys()))

    file_max = epoch_dict[epoch_max]

    return file_max


# 4.1 Some Evaluation Functions


def spearman(y_true, y_pred):

    import pandas as pd

    y_true = y_true.reshape(y_true.shape[0])

    y_pred = y_pred.reshape(y_true.shape[0])

    sp = pd.Series(y_pred).corr(pd.Series(y_true), method="spearman")

    pcc = pd.Series(y_pred).corr(pd.Series(y_true), method="pearson")

    return (sp, pcc)


# 计算 KL Divergence (Kullback-Leibler)


def asymmetricKL(P, Q):
    """

    Epsilon is used here to avoid conditional code for

    checking that neither P nor Q is equal to 0.

    """

    epsilon = 0.00001

    # You may want to instead make copies to avoid changing the np arrays.

    P = P + epsilon

    Q = Q + epsilon

    P_sum = sum(P)

    Q_sum = sum(Q)

    P = P / P_sum

    Q = Q / Q_sum

    divergence = np.sum(P * np.log(P / Q))

    return divergence


def symmetricalKL(P, Q):
    """

    P = np.asarray([1.346112,1.337432,1.246655])

    Q = np.asarray([1.033836,1.082015,1.117323])

    print(asymmetricKL(P, Q), asymmetricKL(Q, P))

    print(symmetricalKL(P, Q))

    """

    return (asymmetricKL(P, Q) + asymmetricKL(Q, P)) / 2.00


# Evaluation

# Metrics: pearson & symmetricKL


def evaluation_repair_map(y_train, Y_pred):

    import numpy as np

    from sklearn.metrics import mean_squared_error

    evaluation_dict = {"pearson": [], "symKL": [], "MSE": []}

    sample_n = y_train.shape[0]

    for index in range(sample_n):

        temp_train = y_train[index, :]

        temp_pred = Y_pred[index, :]

        # pearson

        pccs = np.corrcoef(temp_train, temp_pred)[0, 1]

        evaluation_dict["pearson"].append(pccs)

        # symmetricKL

        symKL = symmetricalKL(temp_train, temp_pred)

        evaluation_dict["symKL"].append(symKL)

        # MSE

        mse = mean_squared_error(temp_train, temp_pred)

        evaluation_dict["MSE"].append(mse)

    # DataFrame

    eval_df = pd.DataFrame(evaluation_dict)

    return eval_df


# evaluation: KL-Divergence, MSE, Pearson


def evaluation(model_path, Xdata, ydata):

    # load model

    from keras.models import load_model

    model = load_model(
        model_path,
        custom_objects={"my_categorical_crossentropy_2": my_categorical_crossentropy_2},
    )

    # predict

    ypred = model.predict(Xdata)

    result = evaluation_repair_map(ydata, ypred)

    return result


# Algorithm

# 自定义损失函数


def my_categorical_crossentropy_2(labels, logits):

    import tensorflow as tf

    """ 

    label = tf.constant([[0,0,1,0,0]], dtype=tf.float32) 

    logit = tf.constant([[-1.2, 2.3, 4.1, 3.0, 1.4]], dtype=tf.float32) 

    logits = tf.nn.softmax(logit) # 计算softmax 

    my_result1 = my_categorical_cross_entropy(labels=label, logits=logits) 

    my_result2 = my_categorical_crossentropy_1(label, logits) 

    my_result3 = my_categorical_crossentropy_2(label, logits) 

    my_result1, my_result2, my_result3 

    """

    return tf.keras.losses.categorical_crossentropy(labels, logits)


# For Predicting the ratio of insertions to deletions


def LRModel_Predicting_Ratio_insertion_to_deletion(X_train):

    from keras.models import Sequential

    from keras.layers import Dense, Dropout, Activation, Flatten

    # 构建神经网络模型

    model = Sequential()

    model.add(Dense(input_dim=X_train.shape[1], units=1))

    model.add(Activation("sigmoid"))

    return model


# For Predicting DSB Repair Map


def LRModel_Predicting_DSB_Repair_Map(X_train, y_train):

    from keras.models import Sequential

    from keras.layers import Dense, Dropout, Activation, Flatten

    # 构建神经网络模型

    model = Sequential()

    model.add(Dense(input_dim=X_train.shape[1], units=y_train.shape[1]))

    model.add(Activation("softmax"))

    return model


# For one model fitting


def model_fitting(
    Id, Xtrain, ytrain, Xval, yval, params, early_stopping_patience, save_dir
):

    import time

    save_params_dir = save_dir + "/%s" % (Id)

    mkdir(save_params_dir)

    start = time.time()

    # model framework

    model = LRModel_Predicting_DSB_Repair_Map(Xtrain, ytrain)

    # compile

    model.compile(loss=my_categorical_crossentropy_2, optimizer=params["optimizer"])

    # checkpoint

    from keras.callbacks import ModelCheckpoint

    checkpoint_file = (
        save_params_dir
        + "/model_weights-improvement-{epoch:04d}-train-{loss:.5f}-test-{val_loss:.5f}.hdf5"
    )

    checkpoint = ModelCheckpoint(
        filepath=checkpoint_file,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="min",
    )

    # early stoppping

    from keras.callbacks import EarlyStopping

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=early_stopping_patience, verbose=0, mode="min"
    )

    callbacks_list = [checkpoint, early_stopping]

    # # 加载

    # from keras.models import load_mode

    # model = load_model(filepath)

    try:

        ## fit

        model.fit(
            Xtrain,
            ytrain,
            batch_size=params["batch_size"],
            epochs=params["epochs"],  ## 新加
            validation_data=(Xval, yval),
            callbacks=callbacks_list,
            verbose=0,
        )

    except OSError as e:

        print(e)

        pass

    # record

    model_path = get_best_checkpoint_path(save_params_dir)

    train_eval = evaluation(model_path, Xtrain, ytrain)

    val_eval = evaluation(model_path, Xval, yval)

    train_eval = train_eval.describe()

    train_KL_median = train_eval.loc["50%", "symKL"]

    train_MSE_median = train_eval.loc["50%", "MSE"]

    train_pccs_median = train_eval.loc["50%", "pearson"]

    val_eval = val_eval.describe()

    val_KL_median = val_eval.loc["50%", "symKL"]

    val_MSE_median = val_eval.loc["50%", "MSE"]

    val_pccs_median = val_eval.loc["50%", "pearson"]

    end = time.time()

    run_time = end - start

    print("run_time:", run_time, "Using:", params)

    record_file_path = save_dir + "/summary_models.log"

    text = "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (
        Id,
        run_time,
        str(params),
        train_KL_median,
        val_KL_median,
        train_MSE_median,
        val_MSE_median,
        train_pccs_median,
        val_pccs_median,
        checkpoint_file.split("/")[-1],
    )

    if not os.path.exists(record_file_path):

        cols = "id\tusing time\tparams\ttrain_KL\tval_KL\ttrain_MSE\tval_MSE\ttrain_pearson\tval_pearson\tcheckpoint_file\n"

        with open(record_file_path, "a") as a:

            a.write(cols)

            a.write(text)

    else:

        with open(record_file_path, "a") as a:

            a.write(text)


def main_fit(
    feat_label, Xtrain, ytrain, Xval, yval, save_dir, epoch, early_stopping_patience
):

    ## 拟合模型

    from keras import optimizers

    record_i = 1

    sgd = optimizers.SGD  # (lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    adam = optimizers.Adam

    for optimizer in [sgd, adam]:

        for lr in [0.1, 0.01]:

            for batch_size in [1024, 512, 256, 128, 64]:

                Id = "%s-DSB_Repair_Model_param_%s" % (feat_label, record_i)

                opt = optimizer(lr=lr)

                params = {
                    "optimizer": opt,
                    "batch_size": batch_size,
                    "epochs": epoch,
                    "learning rate": lr,
                }

                model_fitting(
                    Id,
                    Xtrain,
                    ytrain,
                    Xval,
                    yval,
                    params,
                    early_stopping_patience,
                    save_dir,
                )

                record_i += 1


# main_hyperopt modeling analysis


def main_modeling_feature_analysis(
    train_data,
    val_data,
    int_data,
    model_path_pattern,
    save_dir,
    seq_bp,
    feat_list,
    epoch,
    early_stopping_patience,
):

    y_cols = list(int_data["new category"].unique())

    # Get feature set

    print("Step 1: Get feature set ...")

    train_seq_data, train_MH_data, train_eng_data = main_xgb_prediction(
        train_data, int_data, model_path_pattern, seq_bp
    )

    val_seq_data, val_MH_data, val_eng_data = main_xgb_prediction(
        val_data, int_data, model_path_pattern, seq_bp
    )

    train_feat_dict = {
        "seq_feat": train_seq_data,
        "MH_feat": train_MH_data,
        "eng_feat": train_eng_data,
    }

    val_feat_dict = {
        "seq_feat": val_seq_data,
        "MH_feat": val_MH_data,
        "eng_feat": val_eng_data,
    }

    ytrain = np.array(train_data[y_cols])

    yval = np.array(val_data[y_cols])

    if len(feat_list) == 1:

        Tdata = train_feat_dict[feat_list[0]]

        Vdata = val_feat_dict[feat_list[0]]

        Xtrain, Xval = np.array(Tdata.iloc[:, 2:]), np.array(Vdata.iloc[:, 2:])

    else:

        Tdata = pd.concat(
            [train_feat_dict[feat].iloc[:, 2:] for feat in feat_list], axis=1
        )

        Vdata = pd.concat(
            [val_feat_dict[feat].iloc[:, 2:] for feat in feat_list], axis=1
        )

        Xtrain, Xval = np.array(Tdata), np.array(Vdata)

    ## fit

    feat_label = "_".join(feat_list)

    print("Step 2: Modeling ... ...")

    save_model_dir = save_dir + "/model_%s" % (feat_label)

    print("\nmodel label:", save_model_dir)

    print("======================================================")

    mkdir(save_model_dir)

    main_fit(
        feat_label,
        Xtrain,
        ytrain,
        Xval,
        yval,
        save_model_dir,
        epoch,
        early_stopping_patience,
    )

    print("======================================================")

    print("Finish.")


if __name__ == "__main__":

    import sys

    (
        main_path,
        read_train_data_path,
        read_val_data_path,
        read_int_data_path,
        model_path_pattern,
        save_dir,
        seq_bp,
        feat_list,
        epoch,
        early_stopping_patience,
    ) = sys.argv[1:]

    os.chdir(main_path)

    seq_bp = eval(seq_bp)

    feat_list = eval(feat_list)

    epoch = eval(epoch)

    early_stopping_patience = eval(early_stopping_patience)

    # read data

    train_data = pd.read_csv(read_train_data_path, sep="\t")

    val_data = pd.read_csv(read_val_data_path, sep="\t")

    int_data = pd.read_csv(read_int_data_path)

    main_modeling_feature_analysis(
        train_data,
        val_data,
        int_data,
        model_path_pattern,
        save_dir,
        seq_bp,
        feat_list,
        epoch,
        early_stopping_patience,
    )
