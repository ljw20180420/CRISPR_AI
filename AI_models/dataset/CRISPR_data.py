#!/usr/bin/env python

import re
import datasets
from typing import Callable
import torch
import numpy as np
from .utils import gc_content, GetInsertionCount, GetObservation, GetMH

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

_HOMEPAGE = "https://github.com/ljw20180420/CRISPRdata"

_LICENSE = """
MIT License

Copyright (c) 2025 Jingwei Li

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


class CRISPRDataConfig(datasets.BuilderConfig):
    def __init__(
        self,
        ref_filter: Callable | None,
        cut_filter: Callable | None,
        author_filter: Callable | None,
        file_filter: Callable | None,
        test_ratio: float,
        validation_ratio: float,
        seed: int,
        features: datasets.Features,
        ref1len: int,
        ref2len: int,
        random_insert_uplimit: int,
        insert_uplimit: int,
        **kwargs
    ):
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
        random_insert_uplimit: upper limit of random insertion size discriminated in observations.
        insert_uplimit: upper limit of insertion. Insertion longer than insert_uplimit is count in insert_count_long.
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
        self.random_insert_uplimit = random_insert_uplimit
        self.insert_uplimit = insert_uplimit


class CRISPRData(datasets.GeneratorBasedBuilder):
    def __init__(self, **kargs):
        super().__init__(**kargs)
        self.get_insertion_count = GetInsertionCount("ACGT", self.config.insert_uplimit)
        self.get_observations = GetObservation(
            self.config.ref1len,
            self.config.ref2len,
            self.config.random_insert_uplimit,
        )
        self.get_mh = GetMH(self.config.ref1len, self.config.ref2len)

    def get_input(self, ref1, ref2, cut1, cut2, mh_matrix, rep_num, rep_val, model):
        if model != "FOREcasT":
            mh_lens = rep_val.to(torch.int16)
        if model in ["inDelphi", "Lindel", "FOREcasT"]:
            mask = rep_val == 0
            if model != "FOREcasT":
                mh_idxs = rep_num.cumsum(dim=0)[1::2] - 1
            del_lens = (
                torch.arange(cut1, cut1 - len(ref1) - 1, -1, dtype=torch.int16)[None, :]
                + torch.arange(-cut2, len(ref2) - cut2 + 1, dtype=torch.int16)[:, None]
            )[self.diag_indices]
            if model in ["Lindel", "FOREcasT"]:
                dstarts = torch.arange(-cut1, len(ref1) - cut1 + 1, dtype=torch.int16)[
                    None, :
                ].expand(len(ref2) + 1, -1)[self.diag_indices]
                if model == "Lindel":
                    return del_lens, mh_lens, dstarts, mh_idxs
                else:
                    return del_lens, dstarts
            elif model == "inDelphi":
                gt_poss = (
                    torch.arange(-cut2, len(ref2) - cut2 + 1, dtype=torch.int16) + cut1
                )[:, None].expand(-1, len(ref1) + 1)[self.diag_indices]
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
                dim=0,
                index=torch.arange(len(rep_num) // 2).repeat_interleave(rep_num[1::2]),
                src=observations[self.diag_indices][~mask],
            )
            if model != "FOREcasT":
                mh_idxs = rep_num.cumsum(dim=0)[1::2] - 1
            if model in ["Lindel", "FOREcasT"]:
                if model == "Lindel":
                    observations[
                        self.diag_indices[0][~mask], self.diag_indices[1][~mask]
                    ] = 0
                    observations[
                        self.diag_indices[0][mh_idxs], self.diag_indices[1][mh_idxs]
                    ] = counts
                    return observations[self.diag_indices]
                else:
                    observations = observations.to(torch.float32)
                    observations[
                        self.diag_indices[0][~mask], self.diag_indices[1][~mask]
                    ] = (counts / rep_num[1::2]).repeat_interleave(rep_num[1::2])
                    return observations[self.diag_indices]
            elif model == "inDelphi":
                counts = torch.cat([observations[self.diag_indices][mask], counts])
            return counts
        return None

    # trans_funcs
    def trans_func(self, examples):
        (
            ref1s,
            ref2s,
            cut1s,
            cut2s,
            mh_idxs,
            mh_vals,
            mh_idx_align_ref1s,
            mh_idx_align_ref2s,
            ob_idxs,
            ob_vals,
            insert_countss,
            insert_count_longs,
        ) = ([], [], [], [], [], [], [], [], [], [], [], [])
        for ref1, ref2, cuts in zip(
            examples["ref1"], examples["ref2"], examples["cuts"]
        ):
            for cut in cuts:
                # ref
                ref1s.append(ref1)
                ref2s.append(ref2)
                # cut
                cut1s.append(cut["cut1"])
                cut2s.append(cut["cut2"])
                # mh
                mh_matrix, mh_idx_align_ref1, mh_idx_align_ref2, mh_rep_num = (
                    self.get_mh(ref1, ref2, cut["cut1"], cut["cut2"], ext1=0, ext2=0)
                )
                mh_idx = mh_matrix.nonzero()
                mh_idxs.append(mh_idx)
                mh_vals.append(mh_matrix[mh_idx])
                mh_idx_align_ref1s.append(mh_idx_align_ref1)
                mh_idx_align_ref2s.append(mh_idx_align_ref2)
                # observe
                observations = self.get_observations(ref1, ref2, cut)
                observations = self.get_mh.correct_observation(
                    observations, mh_matrix, mh_rep_num
                ).flatten()
                (ob_idx,) = observations.nonzero()
                ob_idxs.append(ob_idx)
                ob_vals.append(observations[ob_idx])
                # insert
                insert_counts, insert_count_long = self.get_insertion_count(
                    ref1, ref2, cut
                )
                insert_countss.append(insert_counts)
                insert_count_longs.append(insert_count_long)
        return {
            "ref1": ref1s,
            "ref2": ref2s,
            "cut1": cut1s,
            "cut2": cut2s,
            "mh_idx": mh_idxs,
            "mh_val": mh_vals,
            "mh_idx_align_ref1": mh_idx_align_ref1s,
            "mh_idx_align_ref2": mh_idx_align_ref2s,
            "random_insert_uplimit": [self.config.random_insert_uplimit]
            * len(examples["ref1"]),
            "ob_idx": ob_idxs,
            "ob_val": ob_vals,
            "insert_count": insert_countss,
            "insert_count_long": insert_count_longs,
        }

    def inDelphi_trans_func(self, examples):
        (
            refs,
            cut_list,
            mh_del_lenss,
            mh_mh_lenss,
            mh_gt_posss,
            mh_gc_fracss,
            mh_countss,
            mhless_countss,
            insert_1bpss,
        ) = ([], [], [], [], [], [], [], [], [])
        for ref1, ref2, cuts in zip(
            examples["ref1"], examples["ref2"], examples["cuts"]
        ):
            for cut in cuts:
                # ref and cut
                cut1, cut2 = cut["cut1"], cut["cut2"]
                refs.append(ref1[:cut1] + ref2[cut2:])
                cut_list.append(cut1)
                # input
                mh_matrix, rep_num, rep_val = self.num2micro_homology(
                    ref1, ref2, cut1, cut2
                )
                del_lens, mh_lens, gt_poss = self.get_input(
                    ref1, ref2, cut1, cut2, mh_matrix, rep_num, rep_val, "inDelphi"
                )
                mask_del = (
                    (del_lens > 0)
                    & (del_lens < self.config.DELLEN_LIMIT)
                    & (gt_poss >= cut1)
                    & (gt_poss - del_lens <= cut1)
                )
                del_lens, mh_lens, gt_poss = (
                    del_lens[mask_del],
                    mh_lens[mask_del],
                    gt_poss[mask_del],
                )
                mask_mh = mh_lens > 0
                mh_del_lenss.append(del_lens[mask_mh])
                mh_gt_posss.append(gt_poss[mask_mh])
                mh_mh_lenss.append(mh_lens[mask_mh])
                mh_gc_fracss.append(
                    [
                        gc_content(refs[-1][gt_pos - mh_len : gt_pos])
                        for mh_len, gt_pos in zip(mh_mh_lenss[-1], mh_gt_posss[-1])
                    ]
                )
                # output
                observations = self.get_observations(ref1, ref2, cut)
                insert_counts, _ = self.get_insertion_count(ref1, ref2, cut)
                insert_counts = insert_counts[:4]
                insert_1bpss.append(insert_counts)
                counts = self.get_output(observations, rep_num, rep_val, "inDelphi")
                counts = counts[mask_del]
                mh_countss.append(counts[mask_mh])
                mhless_counts = torch.zeros(
                    self.config.DELLEN_LIMIT - 1, dtype=torch.int64
                )
                mhless_counts = mhless_counts.scatter_add(
                    dim=0,
                    index=(del_lens[~mask_mh] - 1).to(torch.int64),
                    src=counts[~mask_mh],
                )
                mhless_countss.append(mhless_counts)
        return {
            "ref": refs,
            "cut": cut_list,
            "mh_gt_pos": mh_gt_posss,
            "mh_del_len": mh_del_lenss,
            "mh_mh_len": mh_mh_lenss,
            "mh_gc_frac": mh_gc_fracss,
            "mh_count": mh_countss,
            "mhless_count": mhless_countss,
            "insert_1bp": insert_1bpss,
        }

    def Lindel_trans_func(self, examples):
        (
            refs,
            cut_list,
            del_counts,
            ins_counts,
            dstarts_list,
            del_lens_list,
            mh_lens_list,
        ) = ([], [], [], [], [], [], [])
        for ref1, ref2, cuts in zip(
            examples["ref1"], examples["ref2"], examples["cuts"]
        ):
            for cut in cuts:
                # ref and cut
                cut1, cut2 = cut["cut1"], cut["cut2"]
                refs.append(ref1[:cut1] + ref2[cut2:])
                cut_list.append(cut1)
                # input
                mh_matrix, rep_num, rep_val = self.num2micro_homology(
                    ref1, ref2, cut1, cut2, ext1=2, ext2=1
                )
                del_lens, mh_lens, dstarts, mh_idxs = self.get_input(
                    ref1, ref2, cut1, cut2, mh_matrix, rep_num, rep_val, "Lindel"
                )
                mask_del_len = (
                    (del_lens > 0)
                    .logical_and(del_lens < self.config.Lindel_dlen)
                    .logical_and(dstarts < 3)
                    .logical_and(dstarts + del_lens > -2)
                )
                mask_mh_end = torch.full(mask_del_len.shape, False)
                mask_mh_end[mh_idxs] = True
                mask = mask_del_len.logical_and((mh_lens == 0).logical_or(mask_mh_end))
                mh_lens = mh_lens[mask]
                dstarts = dstarts[mask]
                dstarts[(dstarts > 0).logical_and(dstarts <= mh_lens)] = 0
                del_lens = del_lens[mask]
                mh_lens = torch.min(del_lens, mh_lens).clamp(
                    0, self.config.Lindel_mh_len
                )
                dstarts_list.append(dstarts)
                del_lens_list.append(del_lens)
                mh_lens_list.append(mh_lens)
                # output
                observations = self.get_observations(ref1, ref2, cut)
                insert_counts, insert_count_long = self.get_insertion_count(
                    ref1, ref2, cut
                )
                ins_counts.append(np.append(insert_counts, insert_count_long))
                counts = self.get_output(observations, rep_num, rep_val, "Lindel")
                del_counts.append(counts[mask_del_len])
        return {
            "ref": refs,
            "cut": cut_list,
            "del_count": del_counts,
            "ins_count": ins_counts,
            "dstart": dstarts_list,
            "del_len": del_lens_list,
            "mh_len": mh_lens_list,
        }

    def FOREcasT_trans_func(self, examples):
        refs, cut_list, total_counts = [], [], []
        for ref1, ref2, cuts in zip(
            examples["ref1"], examples["ref2"], examples["cuts"]
        ):
            for cut in cuts:
                # ref and cut
                cut1, cut2 = cut["cut1"], cut["cut2"]
                refs.append(ref1[:cut1] + ref2[cut2:])
                cut_list.append(cut["cut1"])
                # input
                mh_matrix, rep_num, rep_val = self.num2micro_homology(
                    ref1, ref2, cut1, cut2
                )
                del_lens, dstarts = self.get_input(
                    ref1, ref2, cut1, cut2, mh_matrix, rep_num, rep_val, "FOREcasT"
                )
                mask_del_len = (
                    (del_lens >= 0)
                    .logical_and(del_lens <= self.config.FOREcasT_MAX_DEL_SIZE)
                    .logical_and(dstarts <= 0)
                    .logical_and(dstarts + del_lens >= 0)
                )
                # output
                observations = self.get_observations(ref1, ref2, cut)
                insert_counts, _ = self.get_insertion_count(ref1, ref2, cut)

                counts = self.get_output(observations, rep_num, rep_val, "FOREcasT")
                total_counts.append(torch.cat([counts[mask_del_len], insert_counts]))
        return {"ref": refs, "cut": cut_list, "count": total_counts}

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('path_to_CRISPR_data', 'config_name')

    features = datasets.Features(
        {
            "ref1": datasets.Value("string"),
            "ref2": datasets.Value("string"),
            "cut1": datasets.Value("int64"),
            "cut2": datasets.Value("int64"),
            "mh_idx": datasets.Sequence(datasets.Value("int64")),
            "mh_val": datasets.Sequence(datasets.Value("int64")),
            "mh_idx_align_ref1": datasets.Sequence(datasets.Value("int64")),
            "mh_idx_align_ref2": datasets.Sequence(datasets.Value("int64")),
            "random_insert_uplimit": datasets.Value("int64"),
            "ob_idx": datasets.Sequence(datasets.Value("int64")),
            "ob_val": datasets.Sequence(datasets.Value("float64")),
            "insert_count": datasets.Sequence(datasets.Value("int64")),
            "insert_count_long": datasets.Value("int64"),
        }
    )

    VERSION = datasets.Version("1.0.1")

    BUILDER_CONFIG_CLASS = CRISPRDataConfig

    BUILDER_CONFIGS = [
        CRISPRDataConfig(
            ref_filter=None,
            cut_filter=None,
            author_filter=lambda author, ref1, ref2, cut1, cut2: author == "SX",
            file_filter=lambda file, ref1, ref2, cut1, cut2, author: bool(
                re.search("^(A2-|A7-|D2-)", file)
            ),
            test_ratio=0.05,
            validation_ratio=0.05,
            seed=63036,
            features=features,
            ref1len=127,
            ref2len=127,
            random_insert_uplimit=0,
            insert_uplimit=2,
            name="SX_spcas9",
            version=VERSION,
            description="Data of spcas9",
        ),
        CRISPRDataConfig(
            ref_filter=None,
            cut_filter=None,
            author_filter=lambda author, ref1, ref2, cut1, cut2: author == "SX",
            file_filter=lambda file, ref1, ref2, cut1, cut2, author: bool(
                re.search("^(X-|x-|B2-|36t-)", file)
            ),
            test_ratio=0.05,
            validation_ratio=0.05,
            seed=63036,
            features=features,
            ref1len=127,
            ref2len=127,
            random_insert_uplimit=0,
            insert_uplimit=2,
            name="SX_spymac",
            version=VERSION,
            description="Data of spymac",
        ),
        CRISPRDataConfig(
            ref_filter=None,
            cut_filter=None,
            author_filter=lambda author, ref1, ref2, cut1, cut2: author == "SX",
            file_filter=lambda file, ref1, ref2, cut1, cut2, author: bool(
                re.search("^(i10t-|i83-)", file)
            ),
            test_ratio=0.05,
            validation_ratio=0.05,
            seed=63036,
            features=features,
            ref1len=127,
            ref2len=127,
            random_insert_uplimit=0,
            insert_uplimit=2,
            name="SX_ispymac",
            version=VERSION,
            description="Data of ispymac",
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
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            # citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive

        downloaded_files = dl_manager.download(
            "https://github.com/ljw20180420/CRISPRdata/raw/refs/heads/main/dataset.json.gz"
        )
        # downloaded_files = dl_manager.download("./test.json.gz")

        ds = datasets.load_dataset(
            "json",
            data_files=downloaded_files,
            features=datasets.Features(
                {
                    "ref1": datasets.Value("string"),
                    "ref2": datasets.Value("string"),
                    "cuts": [
                        datasets.Features(
                            {
                                "cut1": datasets.Value("int64"),
                                "cut2": datasets.Value("int64"),
                                "authors": [
                                    datasets.Features(
                                        {
                                            "author": datasets.Value("string"),
                                            "files": [
                                                datasets.Features(
                                                    {
                                                        "file": datasets.Value(
                                                            "string"
                                                        ),
                                                        "ref1_end": datasets.Sequence(
                                                            datasets.Value("int64")
                                                        ),
                                                        "ref2_start": datasets.Sequence(
                                                            datasets.Value("int64")
                                                        ),
                                                        "random_insert": datasets.Sequence(
                                                            datasets.Value("string")
                                                        ),
                                                        "count": datasets.Sequence(
                                                            datasets.Value("int64")
                                                        ),
                                                    }
                                                )
                                            ],
                                        }
                                    )
                                ],
                            }
                        )
                    ],
                }
            ),
        )
        ds = ds.map(self.filter_refs, batched=True)
        ds = ds.map(
            self.trans_func,
            batched=True,
            remove_columns=["cuts"],
        )
        ds = self.split_train_valid_test(ds)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "dataset": ds["train"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "dataset": ds["validation"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "dataset": ds["test"],
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, dataset):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        for id, example in enumerate(dataset):
            yield id, example

    def split_train_valid_test(self, ds):
        # Divide ds's train split to validation and test splits.
        ds = ds["train"].train_test_split(
            test_size=self.config.test_ratio + self.config.validation_ratio,
            shuffle=True,
            seed=self.config.seed,
        )
        ds_valid_test = ds["test"].train_test_split(
            test_size=self.config.test_ratio
            / (self.config.test_ratio + self.config.validation_ratio),
            shuffle=False,
        )
        ds["validation"] = ds_valid_test.pop("train")
        ds["test"] = ds_valid_test.pop("test")
        return ds

    def filter_refs(self, examples):
        ref1s, ref2s, cutss = [], [], []
        for ref1, ref2, cuts in zip(
            examples["ref1"], examples["ref2"], examples["cuts"]
        ):
            if self.config.ref_filter is None or self.config.ref_filter(ref1, ref2):
                if (
                    self.config.cut_filter is not None
                    or self.config.author_filter is not None
                    or self.config.file_filter is not None
                ):
                    cuts = self.filter_cuts(cuts, ref1, ref2)
                if cuts:
                    ref1s.append(ref1)
                    ref2s.append(ref2)
                    cutss.append(cuts)
        return {"ref1": ref1s, "ref2": ref2s, "cuts": cutss}

    def filter_cuts(self, cuts, ref1, ref2):
        new_cuts = []
        for cut in cuts:
            if self.config.cut_filter is None or self.config.cut_filter(
                cut["cut1"], cut["cut2"], ref1, ref2
            ):
                if (
                    self.config.author_filter is not None
                    or self.config.file_filter is not None
                ):
                    cut["authors"] = self.filter_authors(
                        cut["authors"], ref1, ref2, cut["cut1"], cut["cut2"]
                    )
                if cut["authors"]:
                    new_cuts.append(cut)
        return new_cuts

    def filter_authors(self, authors, ref1, ref2, cut1, cut2):
        new_authors = []
        for author in authors:
            if self.config.author_filter is None or self.config.author_filter(
                author["author"], ref1, ref2, cut1, cut2
            ):
                if self.config.file_filter is not None:
                    author["files"] = self.filter_files(
                        author["files"], ref1, ref2, cut1, cut2, author["author"]
                    )
                if author["files"]:
                    new_authors.append(author)
        return new_authors

    def filter_files(self, files, ref1, ref2, cut1, cut2, author):
        return [
            file
            for file in files
            if self.config.file_filter(file["file"], ref1, ref2, cut1, cut2, author)
        ]
