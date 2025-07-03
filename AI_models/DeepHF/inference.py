import torch
import logging
import pathlib
from datasets import load_dataset, Features, Value
from torch.utils.data import DataLoader
from diffusers import DiffusionPipeline
from tqdm import tqdm
import pandas as pd
import tempfile
import subprocess
from .model import DeepHFConfig
from .load_data import data_collator, get_energy, SeqTokenizer


@torch.no_grad()
def data_collator_inference(
    examples: list[dict],
    scaffold: str,
    ext1_up: int,
    ext1_down: int,
    ext2_up: int,
    ext2_down: int,
) -> dict:
    _, tmp_sgRNA = tempfile.mkstemp(text=True)
    _, tmp_sgRNA_scaffold = tempfile.mkstemp(text=True)
    _, tmp_RNAfold_sgRNA = tempfile.mkstemp(text=True)
    _, tmp_RNAfold_sgRNA_scaffold = tempfile.mkstemp(text=True)
    with open(tmp_sgRNA, "w") as sgRNA_fd, open(
        tmp_sgRNA_scaffold, "w"
    ) as sgRNA_scaffold_fd:
        for example in examples:
            ref, cut = example.pop("ref"), example.pop("cut")
            assert (
                cut >= 17 and len(ref) - cut >= 4
            ), f"ref is too short to contain 21mer"
            example["ref1"], example["cut1"] = ref, cut
            sgRNA21mer = ref[cut - 17 : cut + 4]
            sgRNA_fd.write(f">{sgRNA21mer}\n{sgRNA21mer[:20]}\n")
            sgRNA_scaffold_fd.write(f">{sgRNA21mer}\n{sgRNA21mer[:20]}{scaffold}\n")
    subprocess.run(
        f"RNAfold --noPS < {tmp_sgRNA} > {tmp_RNAfold_sgRNA}",
        shell=True,
    )
    subprocess.run(
        f"RNAfold --noPS < {tmp_sgRNA_scaffold} > {tmp_RNAfold_sgRNA_scaffold}",
        shell=True,
    )
    return data_collator(
        examples,
        ext1_up=ext1_up,
        ext1_down=ext1_down,
        ext2_up=ext2_up,
        ext2_down=ext2_down,
        energy_records=get_energy(
            [tmp_RNAfold_sgRNA],
            [tmp_RNAfold_sgRNA_scaffold],
        ),
        seq_tokenizer=SeqTokenizer("PSACGT"),
        output_observation=False,
    )


@torch.no_grad()
def inference(
    data_name: str,
    inference_data: pathlib.Path,
    scaffold: str,
    inference_output: pathlib.Path,
    owner: str,
    batch_size: int,
    device: str,
    logger: logging.Logger,
) -> None:
    logger.info("load inference data")
    ds = load_dataset(
        "json",
        data_files=inference_data,
        features=Features({"ref": Value("string"), "cut": Value("int64")}),
    )["train"]

    logger.info("load pipeline")
    pipe = DiffusionPipeline.from_pretrained(
        f"{owner}/DeepHF_DeepHF_{data_name}",
        trust_remote_code=True,
        custom_pipeline=f"{owner}/DeepHF_DeepHF_{data_name}",
    )
    pipe.DeepHF_model.to(device)

    inference_dataloader = DataLoader(
        dataset=ds,
        batch_size=batch_size,
        collate_fn=lambda examples, scaffold=scaffold, ext1_up=pipe.DeepHF_model.config.ext1_up, ext1_down=pipe.DeepHF_model.config.ext1_down, ext2_up=pipe.DeepHF_model.config.ext2_up, ext2_down=pipe.DeepHF_model.config.ext2_down: data_collator_inference(
            examples,
            scaffold=scaffold,
            ext1_up=ext1_up,
            ext1_down=ext1_down,
            ext2_up=ext2_up,
            ext2_down=ext2_down,
        ),
    )

    dfs = []
    for batch in tqdm(inference_dataloader):
        dfs.append(pd.DataFrame(pipe(batch)))

    pd.concat(dfs).to_csv(inference_output, index=False)
