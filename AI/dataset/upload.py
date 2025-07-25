#!/usr/bin/env python

# path/to/upload.py [test]

from huggingface_hub import HfApi
import pathlib
from datasets.commands.test import TestCommand
import os
import sys

# Change to the directory of the script
os.chdir(pathlib.Path(__file__).parent)


def test():
    test_command = TestCommand(
        dataset="CRISPR_data.py",
        name=None,
        cache_dir=None,
        data_dir=None,
        all_configs=True,
        save_infos=True,
        ignore_verifications=False,
        force_redownload=False,
        clear_cache=False,
        num_proc=None,
        trust_remote_code=True,
    )
    test_command.run()


if __name__ == "__main__":
    owner = os.environ["CRISPR_AI_DATASET__OWNER"]
    api = HfApi()
    api.create_repo(
        repo_id=f"{owner}/CRISPR_data",
        repo_type="dataset",
        exist_ok=True,
    )

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # test() always appends contents to README.md, so remove it before new test().
        if os.path.exists("README.md"):
            os.remove("README.md")
        test()
        api.upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md",
            repo_id=f"{owner}/CRISPR_data",
            repo_type="dataset",
        )

    api.upload_file(
        path_or_fileobj="CRISPR_data.py",
        path_in_repo="CRISPR_data.py",
        repo_id=f"{owner}/CRISPR_data",
        repo_type="dataset",
    )

    api.upload_file(
        path_or_fileobj="utils.py",
        path_in_repo="utils.py",
        repo_id=f"{owner}/CRISPR_data",
        repo_type="dataset",
    )
