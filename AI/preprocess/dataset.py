from datasets import load_dataset
from typing import Literal, Optional
from .generator import MyGenerator


class MyDataset:
    def __init__(
        self,
        user: str,
        repo: str,
        name: Literal["SX_spcas9", "SX_spymac", "SX_ispymac"],
        test_ratio: float,
        validation_ratio: float,
        random_insert_uplimit: int,
        insert_uplimit: int,
        my_generator: Optional[MyGenerator],
    ) -> None:
        """Parameters of dataset.

        Args:
            user: huggingface user name.
            repo: huggingface repo name.
            name: Data name. Generally correpond to Cas protein name.
            test_ratio: Proportion for test samples.
            validation_ratio: Proportion for validation samples.
            random_insert_uplimit: The maximal discriminated length of random insertion.
            insert_uplimit: The maximal insertion length to count.
        """
        self.user = user
        self.repo = repo
        self.name = name
        self.test_ratio = test_ratio
        self.validation_ratio = validation_ratio
        self.random_insert_uplimit = random_insert_uplimit
        self.insert_uplimit = insert_uplimit
        if my_generator is not None:
            self.dataset = load_dataset(
                path=f"{self.user}/{self.repo}",
                name=self.name,
                trust_remote_code=True,
                test_ratio=self.test_ratio,
                validation_ratio=self.validation_ratio,
                random_insert_uplimit=self.random_insert_uplimit,
                insert_uplimit=self.insert_uplimit,
                generator=my_generator.np_rng,
            )
