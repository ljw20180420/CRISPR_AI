#!/usr/bin/env python

import logging
import pathlib
import os
import shutil
from tqdm import tqdm
from datasets import load_dataset
import datasets
import torch
from torch.utils.data import DataLoader
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from .model import DeepHFModel
from .pipeline import DeepHFPipeline
from .load_data import data_collator, get_energy, SeqTokenizer


@torch.no_grad()
def test_DeepHF(
    data_name: str,
    test_ratio: float,
    validation_ratio: float,
    ref1len: int,
    ref2len: int,
    random_insert_uplimit: int,
    insert_uplimit: int,
    owner: str,
    batch_size: int,
    output_dir: pathlib.Path,
    device: str,
    seed: int,
    logger: logging.Logger,
) -> None:
    logger.info("load model")
    DeepHF_model = DeepHFModel.from_pretrained(output_dir / "DeepHF/DeepHF" / data_name)
    # remove parent module name
    DeepHF_model.__module__ = DeepHF_model.__module__.split(".")[-1]

    logger.info("setup pipeline")
    pipe = DeepHFPipeline(DeepHF_model)
    pipe.DeepHF_model.to(device)

    logger.info("load test data")
    ds = load_dataset(
        path=f"{owner}/CRISPR_data",
        name=data_name,
        split=datasets.Split.TEST,
        trust_remote_code=True,
        test_ratio=test_ratio,
        validation_ratio=validation_ratio,
        seed=seed,
        ref1len=ref1len,
        ref2len=ref2len,
        random_insert_uplimit=random_insert_uplimit,
        insert_uplimit=insert_uplimit,
    )
    test_dataloader = DataLoader(
        dataset=ds,
        batch_size=batch_size,
        collate_fn=lambda examples, ext1_up=DeepHF_model.config.ext1_up, ext1_down=DeepHF_model.config.ext1_down, ext2_up=DeepHF_model.config.ext2_up, ext2_down=DeepHF_model.config.ext2_down, energy_records=get_energy(
            [
                "AI_models/DeepHF/rnafold/final_hgsgrna_libb_all_0811_NAA_scaffold_nbt_A1.csv.rnafold.sgRNA",
                "AI_models/DeepHF/rnafold/final_hgsgrna_libb_all_0811_NAA_scaffold_nbt_A2.csv.rnafold.sgRNA",
                "AI_models/DeepHF/rnafold/final_hgsgrna_libb_all_0811_NAA_scaffold_nbt_A3.csv.rnafold.sgRNA",
                "AI_models/DeepHF/rnafold/final_hgsgrna_libb_all_0811_NGG_scaffold_nor_G1.csv.rnafold.sgRNA",
                "AI_models/DeepHF/rnafold/final_hgsgrna_libb_all_0811_NGG_scaffold_nor_G2.csv.rnafold.sgRNA",
                "AI_models/DeepHF/rnafold/final_hgsgrna_libb_all_0811_NGG_scaffold_nor_G3.csv.rnafold.sgRNA",
            ],
            [
                "AI_models/DeepHF/rnafold/final_hgsgrna_libb_all_0811_NAA_scaffold_nbt_A1.csv.rnafold.sgRNA+scaffold",
                "AI_models/DeepHF/rnafold/final_hgsgrna_libb_all_0811_NAA_scaffold_nbt_A2.csv.rnafold.sgRNA+scaffold",
                "AI_models/DeepHF/rnafold/final_hgsgrna_libb_all_0811_NAA_scaffold_nbt_A3.csv.rnafold.sgRNA+scaffold",
                "AI_models/DeepHF/rnafold/final_hgsgrna_libb_all_0811_NGG_scaffold_nor_G1.csv.rnafold.sgRNA+scaffold",
                "AI_models/DeepHF/rnafold/final_hgsgrna_libb_all_0811_NGG_scaffold_nor_G2.csv.rnafold.sgRNA+scaffold",
                "AI_models/DeepHF/rnafold/final_hgsgrna_libb_all_0811_NGG_scaffold_nor_G3.csv.rnafold.sgRNA+scaffold",
            ],
        ), seq_tokenizer=SeqTokenizer(
            "PSACGT"
        ): data_collator(
            examples,
            ext1_up,
            ext1_down,
            ext2_up,
            ext2_down,
            energy_records,
            seq_tokenizer,
            True,
        ),
    )

    logger.info("test pipeline")
    loss, sample_num = 0.0, 0.0
    for batch in tqdm(test_dataloader):
        output = pipe(batch)
        loss += output["loss"]
        sample_num += output["sample_num"]
    loss /= sample_num
    os.makedirs(f"DeepHF/pipeline/DeepHF/{data_name}", exist_ok=True)
    with open(f"DeepHF/pipeline/DeepHF/{data_name}/test_loss", "w") as fd:
        fd.write(f"{loss}\n")

    logger.info("save pipeline")
    pipe.save_pretrained(save_directory=f"DeepHF/pipeline/DeepHF/{data_name}")

    def ignore_func(src, names):
        return [
            name
            for name in names
            if name.startswith(f"{PREFIX_CHECKPOINT_DIR}-") or name.startswith("_")
        ]

    shutil.copyfile(
        "DeepHF/pipeline.py", f"DeepHF/pipeline/DeepHF/{data_name}/pipeline.py"
    )

    for component in pipe.components.keys():
        shutil.copytree(
            output_dir / "DeepHF/DeepHF" / data_name,
            f"DeepHF/pipeline/DeepHF/{data_name}/{component}",
            ignore=ignore_func,
            dirs_exist_ok=True,
        )
