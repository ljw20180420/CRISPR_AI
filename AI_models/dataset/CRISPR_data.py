#!/usr/bin/env python
# A mixture of data from sx, lcy, sj, ljh, lier

import re
import datasets
import re._compiler
from typing import Callable
import os
import torch
import torch.nn.functional as F
from itertools import product

# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
# _CITATION = """\
# @InProceedings{huggingface:dataset,
# title = {A great new dataset},
# author={huggingface, Inc.
# },
# year={2020}
# }
# """

# TODO: Add a link to an official homepage for the dataset here
# _HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
# _LICENSE = ""

def gc_content(seq):
    return (seq.count("G") + seq.count("C")) / len(seq)

class CRISPRDataConfig(datasets.BuilderConfig):
    def __init__(self, ref_filter: Callable | None = None, cut_filter: Callable | None = None, author_filter: Callable | None = None, file_filter: Callable | None = None, test_ratio: float = 0.05, validation_ratio: float = 0.05, seed: int = 63036, features: datasets.Features = None, ref1len: int = 127, ref2len: int = 127, DELLEN_LIMIT: int = 60, Lindel_dlen: int = 30, Lindel_mh_len: int = 4, FOREcasT_MAX_DEL_SIZE: int = 30, **kwargs):
        """BuilderConfig for CRISPR_data.
        Args:
        trans_func: *function*, transform function applied after filter.
        ref_filter: *function*, ref_filter(ref1, ref2) -> bool.
        cut_filter: *function*, cut_filter(cut1, cut2, ref1[optional], ref2[optional]) -> bool.
        author_filter: *function*, author_filter(author, ref1[optional], ref2[optional], cut1[optional], cut2[optional]) -> bool.
        file_filter: *function*, file_filter(file, ref1[optional], ref2[optional], cut1[optional], cut2[optional], author[optional]) -> bool.
        test_ratio: *float*, the ratio of data for test.
        validation_ratio: *float*, the ratio of data for validation.
        seed: *int*, the random seed.
        features: include the data structure in config (for auto generation of model card when test dataset).
        ref1len: length of ref1.
        ref2len: length of ref2.
        DELLEN_LIMIT: upper limit of inDelphi deletion size.
        Lindel_dlen: upper limit of Lindel deletion size.
        Lindel_mh_len: upper limit of Lindel micro-homology size (mh longer than Lindel_mh_len is not excluded, but cutoff to Lindel_mh_len).
        FOREcasT_MAX_DEL_SIZE: upper limit of FOREcasT deletion size.
        **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(**kwargs)
        self.ref_filter = ref_filter
        self.cut_filter = cut_filter
        self.author_filter = author_filter
        self.file_filter = file_filter
        self.test_ratio = test_ratio
        self.validation_ratio = validation_ratio
        self.seed = seed
        self.features = features
        self.ref1len = ref1len
        self.ref2len = ref2len
        self.DELLEN_LIMIT = DELLEN_LIMIT
        self.Lindel_dlen = Lindel_dlen
        self.Lindel_mh_len = Lindel_mh_len
        self.FOREcasT_MAX_DEL_SIZE = FOREcasT_MAX_DEL_SIZE

class CRISPRData(datasets.GeneratorBasedBuilder):
    def __init__(self, **kargs):
        super().__init__(**kargs)
        with torch.no_grad():
            # diag_indices example for ref2len = 3 and ref1len = 2:
            # 6 9 11   row_indices 0 0 0   col_indices 0 1 2
            # 3 7 10               1 1 1               0 1 2
            # 1 4 8                2 2 2               0 1 2
            # 0 2 5                3 3 3               0 1 2
            # diag_indices = (
            #     tensor([3, 2, 3, 1, 2, 3, 0, 1, 2, 0, 1, 0]),
            #     tensor([0, 0, 1, 0, 1, 2, 0, 1, 2, 1, 2, 2])
            # )
            ref1len, ref2len = self.config.ref1len, self.config.ref2len
            row_indices = torch.arange(ref2len + 1)[:, None].expand(-1, ref1len + 1)
            col_indices = torch.arange(ref1len + 1).expand(ref2len + 1, -1)
            self.diag_indices = (
                # row index
                torch.cat([
                    row_indices.diagonal(offset)
                    for offset in range(-ref2len, ref1len + 1)
                ]),
                # col index
                torch.cat([
                    col_indices.diagonal(offset)
                    for offset in range(-ref2len, ref1len + 1)
                ])
            )
            # 'ACGTN' -> '02143'
            nuc_code = "".join(
                str(i.item())
                for i in torch.frombuffer("ACGTN".encode(), dtype=torch.int8) % 5
            )
            self.base_idx = [
                torch.tensor([
                    int("".join(j), base=5)
                    for j in product(*([nuc_code] * i))
                ])
                for i in range(1, 3)
            ]
            cutoff = 0
            for bi in self.base_idx:
                bi += cutoff
                cutoff += len(bi)
            self.base_idx = torch.cat(self.base_idx)
        

    # auxiliary methods
    def get_observations(self, ref1, ref2, cut, RANDOM_INSERT_LIMIT=99999, INSERT_LIMIT=None, count_long_insertion=False):
        assert len(ref1) == self.config.ref1len and len(ref2) == self.config.ref2len, "reference length does not fit"
        observations = torch.zeros(len(ref2) + 1, len(ref1) + 1, dtype=torch.int64)
        if INSERT_LIMIT is not None:
            base = len("ACGTN")
            total = (base ** (INSERT_LIMIT + 1) - base) // (base - 1)
            assert total <= len(self.base_idx), "INSERT_LIMIT is beyonded"
            insert_counts = torch.zeros(total, dtype=torch.int64)
            if count_long_insertion:
                insert_count_long = 0
            base_vec = base ** torch.arange(INSERT_LIMIT)
            base_cutoff = (base ** (torch.arange(INSERT_LIMIT) + 1) - base) // (base - 1)
            cut1, cut2 = cut['cut1'], cut['cut2']
        for author in cut['authors']:
            for file in author['files']:
                mask = torch.tensor([len(random_insert) <= RANDOM_INSERT_LIMIT for random_insert in file['random_insert']])
                observations[
                    torch.tensor(file['ref2_start'])[mask],
                    torch.tensor(file['ref1_end'])[mask]
                ] += torch.tensor(file['count'])[mask]
                if INSERT_LIMIT is not None:
                    for ref1_end, ref2_start, random_insert, count in zip(file['ref1_end'], file['ref2_start'], file['random_insert'], file['count']):
                        if ref1_end < cut1 or ref2_start > cut2:
                            continue
                        insert = ref1[cut1:ref1_end] + random_insert + ref2[ref2_start:cut2]
                        if len(insert) > 0:
                            if len(insert) <= INSERT_LIMIT:
                                insert_counts[base_cutoff[len(insert) - 1] + (
                                    base_vec[:len(insert)] * (torch.frombuffer(insert.encode(), dtype=torch.int8) % 5)
                                ).sum()] += count
                            elif count_long_insertion:
                                insert_count_long += count
        if INSERT_LIMIT is not None:
            insert_counts = insert_counts[self.base_idx[:total]]
            if count_long_insertion:
                return observations, torch.cat([
                    insert_counts,
                    torch.tensor([insert_count_long], dtype=torch.int64) 
                ])
            else:
                return observations, insert_counts
        return observations

    def num2micro_homology(self, ref1, ref2, cut1, cut2, ext1=0, ext2=0):
        assert len(ref1) == self.config.ref1len and len(ref2) == self.config.ref2len, "reference length does not fit"
        mh_matrix = F.pad(
            (torch.frombuffer(ref1[:cut1 + ext1].encode(), dtype=torch.int8).view(1, -1) == torch.frombuffer(ref2[cut2 - ext2:].encode(), dtype=torch.int8).view(-1, 1)).to(torch.int16),
            pad=(0, len(ref1) - cut1 - ext1 + 1, cut2 - ext2, 1), value=0
        )
        rep_num = torch.cat((
            torch.tensor([-1], dtype=torch.int64),
            torch.where(mh_matrix[self.diag_indices].diff())[0],
            torch.tensor([(len(ref1) + 1) * (len(ref2) + 1) - 1], dtype=torch.int64)
        )).diff()
        rep_val = rep_num.clone()
        rep_val[0::2] = 0
        rep_num[1::2] = rep_num[1::2] + 1
        rep_num[2::2] = rep_num[2::2] - 1
        rep_val = rep_val.repeat_interleave(rep_num)
        return mh_matrix, rep_num, rep_val

    def get_input(self, ref1, ref2, cut1, cut2, mh_matrix, rep_num, rep_val, model):
        if model != "FOREcasT":
            mh_lens = rep_val.to(torch.int16)
        if model in ["inDelphi", "Lindel", "FOREcasT"]:
            mask = rep_val == 0
            if model != "FOREcasT":
                mh_idxs = rep_num.cumsum(dim=0)[1::2] - 1
            del_lens = (torch.arange(cut1, cut1 - len(ref1) - 1, -1, dtype=torch.int16)[None, :] + torch.arange(-cut2, len(ref2) - cut2 + 1, dtype=torch.int16)[:, None])[self.diag_indices]
            if model in ["Lindel", "FOREcasT"]:
                dstarts = torch.arange(-cut1, len(ref1) - cut1 + 1, dtype=torch.int16)[None, :].expand(len(ref2) + 1, -1)[self.diag_indices]
                if model == "Lindel":
                    return del_lens, mh_lens, dstarts, mh_idxs
                else:
                    return del_lens, dstarts
            elif model == "inDelphi":
                gt_poss = (torch.arange(-cut2, len(ref2) - cut2 + 1, dtype=torch.int16) + cut1)[:, None].expand(-1, len(ref1) + 1)[self.diag_indices]
                del_lens = torch.cat([del_lens[mask], del_lens[mh_idxs]])
                gt_poss = torch.cat([gt_poss[mask], gt_poss[mh_idxs]])
                mh_lens = torch.cat([mh_lens[mask], mh_lens[mh_idxs]])
            return del_lens, mh_lens, gt_poss
        mh_matrix[self.diag_indices] = mh_lens
        return mh_matrix

    def get_output(self, observations, rep_num, rep_val, model):
        if model in ["inDelphi", "Lindel", "FOREcasT"]:
            mask = rep_val == 0
            counts = torch.zeros(len(rep_num) // 2, dtype=torch.int64)
            counts = counts.scatter_add(
                dim = 0,
                index = torch.arange(len(rep_num) // 2).repeat_interleave(rep_num[1::2]),
                src = observations[self.diag_indices][~mask]
            )
            if model != "FOREcasT":
                mh_idxs = rep_num.cumsum(dim=0)[1::2] - 1
            if model in ["Lindel", "FOREcasT"]:
                if model == "Lindel":
                    observations[self.diag_indices[0][~mask], self.diag_indices[1][~mask]] = 0
                    observations[self.diag_indices[0][mh_idxs], self.diag_indices[1][mh_idxs]] = counts
                    return observations[self.diag_indices]
                else:
                    observations = observations.to(torch.float32)
                    observations[self.diag_indices[0][~mask], self.diag_indices[1][~mask]] = (counts / rep_num[1::2]).repeat_interleave(rep_num[1::2])
                    return observations[self.diag_indices]
            elif model == "inDelphi":
                counts = torch.cat([observations[self.diag_indices][mask], counts])
            return counts
        return None

    # trans_funcs
    def CRISPR_diffuser_trans_func(self, examples):
        ref1s, ref2s, cut1s, cut2s, mh_ref1s, mh_ref2s, mh_vals, ob_ref1s, ob_ref2s, ob_vals = [], [], [], [], [], [], [], [], [], []
        for ref1, ref2, cuts in zip(examples['ref1'], examples['ref2'], examples['cuts']):
            for cut in cuts:
                # ref and cut
                ref1s.append(ref1)
                ref2s.append(ref2)
                cut1, cut2 = cut['cut1'], cut['cut2']
                cut1s.append(cut1)
                cut2s.append(cut2)
                # input
                mh_matrix, rep_num, rep_val = self.num2micro_homology(ref1, ref2, cut1, cut2)
                mh_matrix = self.get_input(ref1, ref2, cut1, cut2, mh_matrix, rep_num, rep_val, "CRISPR_diffuser")
                mh_ref2, mh_ref1 = mh_matrix.nonzero(as_tuple=True)
                mh_ref1s.append(mh_ref1)
                mh_ref2s.append(mh_ref2)
                mh_vals.append(mh_matrix[mh_ref2, mh_ref1])
                # output
                observations = self.get_observations(ref1, ref2, cut)
                ob_ref2, ob_ref1 = observations.nonzero(as_tuple=True)
                ob_ref1s.append(ob_ref1)
                ob_ref2s.append(ob_ref2)
                ob_vals.append(observations[ob_ref2, ob_ref1])
        return {
            'ref1': ref1s,
            'ref2': ref2s,
            'cut1': cut1s,
            'cut2': cut2s,
            'mh_ref1': mh_ref1s,
            'mh_ref2': mh_ref2s,
            'mh_val': mh_vals,
            'ob_ref1': ob_ref1s,
            'ob_ref2': ob_ref2s,
            'ob_val': ob_vals
        }

    def inDelphi_trans_func(self, examples):
        refs, cut_list, mh_del_lenss, mh_mh_lenss, mh_gt_posss, mh_gc_fracss, mh_countss, mhless_countss, insert_1bpss = [], [], [], [], [], [], [], [], []
        for ref1, ref2, cuts in zip(examples['ref1'], examples['ref2'], examples['cuts']):
            for cut in cuts:
                # ref and cut
                cut1, cut2 = cut['cut1'], cut['cut2']
                refs.append(ref1[:cut1] + ref2[cut2:])
                cut_list.append(cut1)
                # input
                mh_matrix, rep_num, rep_val = self.num2micro_homology(ref1, ref2, cut1, cut2)
                del_lens, mh_lens, gt_poss = self.get_input(ref1, ref2, cut1, cut2, mh_matrix, rep_num, rep_val, "inDelphi")
                mask_del = (del_lens > 0) & (del_lens < self.config.DELLEN_LIMIT) & (gt_poss >= cut1) & (gt_poss - del_lens <= cut1)
                del_lens, mh_lens, gt_poss = del_lens[mask_del], mh_lens[mask_del], gt_poss[mask_del]
                mask_mh = mh_lens > 0
                mh_del_lenss.append(del_lens[mask_mh])
                mh_gt_posss.append(gt_poss[mask_mh])
                mh_mh_lenss.append(mh_lens[mask_mh])
                mh_gc_fracss.append([gc_content(refs[-1][gt_pos - mh_len:gt_pos]) for mh_len, gt_pos in zip(mh_mh_lenss[-1], mh_gt_posss[-1])])
                # output
                observations, insert_counts = self.get_observations(ref1, ref2, cut, INSERT_LIMIT = 1)
                insert_1bpss.append(insert_counts)                
                counts = self.get_output(observations, rep_num, rep_val, "inDelphi")
                counts = counts[mask_del]
                mh_countss.append(counts[mask_mh])
                mhless_counts = torch.zeros(self.config.DELLEN_LIMIT - 1, dtype=torch.int64)
                mhless_counts = mhless_counts.scatter_add(dim = 0, index=(del_lens[~mask_mh] - 1).to(torch.int64), src=counts[~mask_mh])
                mhless_countss.append(mhless_counts)
        return {
            'ref': refs,
            'cut': cut_list,
            'mh_gt_pos': mh_gt_posss,
            'mh_del_len': mh_del_lenss,
            'mh_mh_len': mh_mh_lenss,
            'mh_gc_frac': mh_gc_fracss,
            'mh_count': mh_countss,
            'mhless_count': mhless_countss,
            'insert_1bp': insert_1bpss
        }

    def Lindel_trans_func(self, examples):
        refs, cut_list, del_counts, ins_counts, dstarts_list, del_lens_list, mh_lens_list  = [], [], [], [], [], [], []
        for ref1, ref2, cuts in zip(examples['ref1'], examples['ref2'], examples['cuts']):
            for cut in cuts:
                # ref and cut
                cut1, cut2 = cut['cut1'], cut['cut2']
                refs.append(ref1[:cut1] + ref2[cut2:])
                cut_list.append(cut1)
                # input
                mh_matrix, rep_num, rep_val = self.num2micro_homology(ref1, ref2, cut1, cut2, ext1=2, ext2=1)
                del_lens, mh_lens, dstarts, mh_idxs = self.get_input(ref1, ref2, cut1, cut2, mh_matrix, rep_num, rep_val, "Lindel")
                mask_del_len = (del_lens > 0).logical_and(del_lens < self.config.Lindel_dlen).logical_and(dstarts < 3).logical_and(dstarts + del_lens > -2)
                mask_mh_end = torch.full(mask_del_len.shape, False)
                mask_mh_end[mh_idxs] = True
                mask = mask_del_len.logical_and((mh_lens == 0).logical_or(mask_mh_end))
                mh_lens = mh_lens[mask]
                dstarts = dstarts[mask]
                dstarts[(dstarts > 0).logical_and(dstarts <= mh_lens)] = 0
                del_lens = del_lens[mask]
                mh_lens = torch.min(del_lens, mh_lens).clamp(0, self.config.Lindel_mh_len)
                dstarts_list.append(dstarts)
                del_lens_list.append(del_lens)
                mh_lens_list.append(mh_lens)
                #output
                observations, insert_counts = self.get_observations(ref1, ref2, cut, INSERT_LIMIT=2, count_long_insertion=True)
                insert_counts = insert_counts.tolist()
                del insert_counts[-6:-1]
                del insert_counts[4::5]
                ins_counts.append(insert_counts)
                counts = self.get_output(observations, rep_num, rep_val, "Lindel")               
                del_counts.append(counts[mask_del_len])
        return {
            'ref': refs,
            'cut': cut_list,
            'del_count': del_counts,
            'ins_count': ins_counts,
            'dstart': dstarts_list,
            'del_len': del_lens_list,
            'mh_len': mh_lens_list
        }

    def FOREcasT_trans_func(self, examples):
        refs, cut_list, total_counts = [], [], []
        for ref1, ref2, cuts in zip(examples['ref1'], examples['ref2'], examples['cuts']):
            for cut in cuts:
                # ref and cut
                cut1, cut2 = cut["cut1"], cut["cut2"]
                refs.append(ref1[:cut1] + ref2[cut2:])
                cut_list.append(cut['cut1'])
                # input
                mh_matrix, rep_num, rep_val = self.num2micro_homology(ref1, ref2, cut1, cut2)
                del_lens, dstarts = self.get_input(ref1, ref2, cut1, cut2, mh_matrix, rep_num, rep_val, "FOREcasT")
                mask_del_len = (del_lens >= 0).logical_and(del_lens <= self.config.FOREcasT_MAX_DEL_SIZE).logical_and(dstarts <= 0).logical_and(dstarts + del_lens >= 0)
                # output
                observations, insert_counts = self.get_observations(ref1, ref2, cut, RANDDOM_INSERT_LIMIT=0, INSERT_LIMIT=2)
                insert_counts = insert_counts.tolist()
                del insert_counts[-5:]
                del insert_counts[4::5]
                counts = self.get_output(observations, rep_num, rep_val, "FOREcasT")
                total_counts.append(torch.cat([
                    counts[mask_del_len],
                    torch.tensor(insert_counts)
                ]))
        return {
            'ref': refs,
            'cut': cut_list,
            'count': total_counts
        }

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('path_to_CRISPR_data', 'config_name')

    features_CRISPR_diffuser = datasets.Features({
        'ref1': datasets.Value('string'),
        'ref2': datasets.Value('string'),
        'cut1': datasets.Value('int16'),
        'cut2': datasets.Value('int16'),
        'mh_ref1': datasets.Sequence(datasets.Value('int16')),
        'mh_ref2': datasets.Sequence(datasets.Value('int16')),
        'mh_val': datasets.Sequence(datasets.Value('int16')),
        'ob_ref1': datasets.Sequence(datasets.Value('int16')),
        'ob_ref2': datasets.Sequence(datasets.Value('int16')),
        'ob_val': datasets.Sequence(datasets.Value('int64'))
    })

    features_inDelphi = datasets.Features({
        'ref': datasets.Value('string'),
        'cut': datasets.Value('int16'),
        'mh_gt_pos': datasets.Sequence(datasets.Value('int16')),
        'mh_del_len': datasets.Sequence(datasets.Value('int16')),
        'mh_mh_len': datasets.Sequence(datasets.Value('int16')),
        'mh_gc_frac': datasets.Sequence(datasets.Value('float32')),
        'mh_count': datasets.Sequence(datasets.Value('int64')),
        'mhless_count': datasets.Sequence(datasets.Value('int64')),
        'insert_1bp': datasets.Sequence(datasets.Value('int64'))
    })

    features_Lindel = datasets.Features({
        'ref': datasets.Value('string'),
        'cut': datasets.Value('int16'),
        'del_count': datasets.Sequence(datasets.Value('int64')),
        'ins_count': datasets.Sequence(datasets.Value('int64'))
    })

    features_FOREcasT = datasets.Features({
        'ref': datasets.Value('string'),
        'cut': datasets.Value('int16'),
        'count': datasets.Sequence(datasets.Value('float32'))
    })

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIG_CLASS = CRISPRDataConfig

    BUILDER_CONFIGS = [
        CRISPRDataConfig(
            author_filter = lambda author, ref1, ref2, cut1, cut2: author == "SX",
            file_filter = lambda file, ref1, ref2, cut1, cut2, author: bool(re.search("^(A2-|A7-|D2-)", file)),
            features = features_CRISPR_diffuser,
            name = "SX_spcas9_CRISPR_diffuser",
            version = VERSION,
            description = "Data of spcas9 protein of sx and lcy for CRISPR diffuser training"
        ),
        CRISPRDataConfig(
            author_filter = lambda author, ref1, ref2, cut1, cut2: author == "SX",
            file_filter = lambda file, ref1, ref2, cut1, cut2, author: bool(re.search("^(X-|x-|B2-|36t-)", file)),
            features = features_CRISPR_diffuser,
            name = "SX_spymac_CRISPR_diffuser",
            version = VERSION,
            description = "Data of spymac protein of sx and lcy for CRISPR diffuser training"
        ),
        CRISPRDataConfig(
            author_filter = lambda author, ref1, ref2, cut1, cut2: author == "SX",
            file_filter = lambda file, ref1, ref2, cut1, cut2, author: bool(re.search("^(i10t-|i83-)", file)),
            features = features_CRISPR_diffuser,
            name = "SX_ispymac_CRISPR_diffuser",
            version = VERSION,
            description = "Data of ispymac protein of sx and lcy for CRISPR diffuser training"
        ),
        CRISPRDataConfig(
            author_filter = lambda author, ref1, ref2, cut1, cut2: author == "SX",
            file_filter = lambda file, ref1, ref2, cut1, cut2, author: bool(re.search("^(A2-|A7-|D2-)", file)),
            features = features_inDelphi,
            name = "SX_spcas9_inDelphi",
            version = VERSION,
            description = "Data of spcas9 protein of sx and lcy for inDelphi training"
        ),
        CRISPRDataConfig(
            author_filter = lambda author, ref1, ref2, cut1, cut2: author == "SX",
            file_filter = lambda file, ref1, ref2, cut1, cut2, author: bool(re.search("^(X-|x-|B2-|36t-)", file)),
            features = features_inDelphi,
            name = "SX_spymac_inDelphi",
            version = VERSION,
            description = "Data of spymac protein of sx and lcy for inDelphi training"
        ),
        CRISPRDataConfig(
            author_filter = lambda author, ref1, ref2, cut1, cut2: author == "SX",
            file_filter = lambda file, ref1, ref2, cut1, cut2, author: bool(re.search("^(i10t-|i83-)", file)),
            features = features_inDelphi,
            name = "SX_ispymac_inDelphi",
            version = VERSION,
            description = "Data of ispymac protein of sx and lcy for inDelphi training"
        ),
        CRISPRDataConfig(
            author_filter = lambda author, ref1, ref2, cut1, cut2: author == "SX",
            file_filter = lambda file, ref1, ref2, cut1, cut2, author: bool(re.search("^(A2-|A7-|D2-)", file)),
            features = features_Lindel,
            name = "SX_spcas9_Lindel",
            version = VERSION,
            description = "Data of spcas9 protein of sx and lcy for Lindel training"
        ),
        CRISPRDataConfig(
            author_filter = lambda author, ref1, ref2, cut1, cut2: author == "SX",
            file_filter = lambda file, ref1, ref2, cut1, cut2, author: bool(re.search("^(X-|x-|B2-|36t-)", file)),
            features = features_Lindel,
            name = "SX_spymac_Lindel",
            version = VERSION,
            description = "Data of spymac protein of sx and lcy for Lindel training"
        ),
        CRISPRDataConfig(
            author_filter = lambda author, ref1, ref2, cut1, cut2: author == "SX",
            file_filter = lambda file, ref1, ref2, cut1, cut2, author: bool(re.search("^(i10t-|i83-)", file)),
            features = features_Lindel,
            name = "SX_ispymac_Lindel",
            version = VERSION,
            description = "Data of ispymac protein of sx and lcy for Lindel training"
        ),
        CRISPRDataConfig(
            author_filter = lambda author, ref1, ref2, cut1, cut2: author == "SX",
            file_filter = lambda file, ref1, ref2, cut1, cut2, author: bool(re.search("^(A2-|A7-|D2-)", file)),
            features = features_FOREcasT,
            name = "SX_spcas9_FOREcasT",
            version = VERSION,
            description = "Data of spcas9 protein of sx and lcy for FOREcasT training"
        ),
        CRISPRDataConfig(
            author_filter = lambda author, ref1, ref2, cut1, cut2: author == "SX",
            file_filter = lambda file, ref1, ref2, cut1, cut2, author: bool(re.search("^(X-|x-|B2-|36t-)", file)),
            features = features_FOREcasT,
            name = "SX_spymac_FOREcasT",
            version = VERSION,
            description = "Data of spymac protein of sx and lcy for FOREcasT training"
        ),
        CRISPRDataConfig(
            author_filter = lambda author, ref1, ref2, cut1, cut2: author == "SX",
            file_filter = lambda file, ref1, ref2, cut1, cut2, author: bool(re.search("^(i10t-|i83-)", file)),
            features = features_FOREcasT,
            name = "SX_ispymac_FOREcasT",
            version = VERSION,
            description = "Data of ispymac protein of sx and lcy for FOREcasT training"
        ),
    ]

    # DEFAULT_CONFIG_NAME = "SX_spcas9"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description="""\
                This dataset is used to train a DL model predicting editing results of CRISPR.
            """,
            # This defines the different columns of the dataset and their types
            # features=
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            # homepage=_HOMEPAGE,
            # License for the dataset if available
            # license=_LICENSE,
            # Citation for the dataset
            # citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        hf_endpoint = os.environ.get("HF_ENDPOINT", "https://" + "huggingface.co")
        downloaded_files = dl_manager.download(f"{hf_endpoint}/datasets/ljw20180420/CRISPR_data/resolve/main/dataset.json.gz")
        # downloaded_files = dl_manager.download("./test.json.gz")
        
        ds = datasets.load_dataset('json', data_files=downloaded_files, features=datasets.Features({
            'ref1': datasets.Value('string'),
            'ref2': datasets.Value('string'),
            'cuts': [datasets.Features({
                'cut1': datasets.Value('int16'),
                'cut2': datasets.Value('int16'),
                'authors': [datasets.Features({
                    'author': datasets.Value('string'),
                    'files': [datasets.Features({
                        'file': datasets.Value('string'),
                        'ref1_end': datasets.Sequence(datasets.Value('int16')),
                        'ref2_start': datasets.Sequence(datasets.Value('int16')),
                        'random_insert': datasets.Sequence(datasets.Value('string')),
                        'count': datasets.Sequence(datasets.Value('int64'))
                    })]
                })]
            })]
        }))
        ds = ds.map(self.filter_refs, batched=True)
        if self.config.name.endswith("_CRISPR_diffuser"):
            ds = ds.map(self.CRISPR_diffuser_trans_func, batched=True, remove_columns=['cuts'])
        elif self.config.name.endswith("_inDelphi"):
            ds = ds.map(self.inDelphi_trans_func, batched=True, remove_columns=['ref1', 'ref2', 'cuts'])
        elif self.config.name.endswith("_Lindel"):
            ds = ds.map(self.Lindel_trans_func, batched=True, remove_columns=['ref1', 'ref2', 'cuts'])
        elif self.config.name.endswith("_FOREcasT"):
            ds = ds.map(self.FOREcasT_trans_func, batched=True, remove_columns=['ref1', 'ref2', 'cuts'])
        ds = self.split_train_valid_test(ds)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "dataset": ds['train'],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "dataset": ds['validation'],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "dataset": ds['test'],
                },
            )
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, dataset):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        for id, example in enumerate(dataset):
            yield id, example

    def split_train_valid_test(self, ds):
        # Divide ds's train split to validation and test splits.
        ds = ds['train'].train_test_split(test_size=self.config.test_ratio + self.config.validation_ratio, shuffle=True, seed=self.config.seed)
        ds_valid_test = ds['test'].train_test_split(test_size=self.config.test_ratio / (self.config.test_ratio + self.config.validation_ratio), shuffle=False)
        ds['validation'] = ds_valid_test.pop('train')
        ds['test'] = ds_valid_test.pop('test')
        return ds

    def filter_refs(self, examples):
        ref1s, ref2s, cutss = [], [], []
        for ref1, ref2, cuts in zip(examples['ref1'], examples['ref2'], examples['cuts']):
            if self.config.ref_filter is None or self.config.ref_filter(ref1, ref2):
                if self.config.cut_filter is not None or self.config.author_filter is not None or self.config.file_filter is not None:
                    cuts = self.filter_cuts(cuts, ref1, ref2)
                if cuts:
                    ref1s.append(ref1)
                    ref2s.append(ref2)
                    cutss.append(cuts)
        return {
            "ref1": ref1s,
            "ref2": ref2s,
            "cuts": cutss
        }

    def filter_cuts(self, cuts, ref1, ref2):
        new_cuts = []
        for cut in cuts:
            if self.config.cut_filter is None or self.config.cut_filter(cut["cut1"], cut["cut2"], ref1, ref2):
                if self.config.author_filter is not None or self.config.file_filter is not None:
                    cut["authors"] = self.filter_authors(cut["authors"], ref1, ref2, cut["cut1"], cut["cut2"])
                if cut["authors"]:
                    new_cuts.append(cut)
        return new_cuts

    def filter_authors(self, authors, ref1, ref2, cut1, cut2):
        new_authors = []
        for author in authors:
            if self.config.author_filter is None or self.config.author_filter(author["author"], ref1, ref2, cut1, cut2):
                if self.config.file_filter is not None:
                    author["files"] = self.filter_files(author["files"], ref1, ref2, cut1, cut2, author["author"])
                if author["files"]:
                    new_authors.append(author)
        return new_authors
    
    def filter_files(self, files, ref1, ref2, cut1, cut2, author):
        return [file for file in files if self.config.file_filter(file["file"], ref1, ref2, cut1, cut2, author)]
