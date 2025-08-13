import datasets
from typing import Literal
import numpy as np


def get_dataset(
    user: Literal["ljw20180420"],
    repo: Literal["CRISPR_data"],
    name: Literal["SX_spcas9", "SX_spymac", "SX_ispymac"],
    test_ratio: float,
    validation_ratio: float,
    random_insert_uplimit: int,
    insert_uplimit: int,
    seed: int,
) -> datasets.Dataset:
    """Parameters of dataset.

    Args:
        user: huggingface user name.
        repo: huggingface repo name.
        name: Data name. Generally correpond to Cas protein name.
        test_ratio: Proportion for test samples.
        validation_ratio: Proportion for validation samples.
        random_insert_uplimit: The maximal discriminated length of random insertion.
        insert_uplimit: The maximal insertion length to count.
        seed: random seed.
    """
    return datasets.load_dataset(
        path=f"{user}/{repo}",
        name=name,
        trust_remote_code=True,
        test_ratio=test_ratio,
        validation_ratio=validation_ratio,
        random_insert_uplimit=random_insert_uplimit,
        insert_uplimit=insert_uplimit,
        generator=np.random.default_rng(seed=seed),
    )
