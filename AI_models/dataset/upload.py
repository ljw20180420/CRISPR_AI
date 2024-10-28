from huggingface_hub import HfApi
from pathlib import Path
from ..proxy import *

def upload():
    api = HfApi()
    api.create_repo(
        repo_id="ljw20180420/CRISPR_data",
        repo_type="dataset",
        exist_ok=True
    )
    api.upload_file(
        path_or_fileobj=Path(__file__).parent / "CRISPR_data.py",
        path_in_repo="CRISPR_data.py",
        repo_id="ljw20180420/CRISPR_data",
        repo_type="dataset"
    )
    api.upload_file(
        path_or_fileobj=Path(__file__).parent / "dataset.json.gz",
        path_in_repo="dataset.json.gz",
        repo_id="ljw20180420/CRISPR_data",
        repo_type="dataset"
    )

    # from datasets.commands.test import TestCommand
    # test_command = TestCommand(
    #     dataset="dataset/CRISPR_data.py",
    #     name=None,
    #     cache_dir=None,
    #     data_dir=None,
    #     all_configs=True,
    #     save_infos=True,
    #     ignore_verifications=False,
    #     force_redownload=False,
    #     clear_cache=False,
    #     num_proc=None,
    #     trust_remote_code=True
    # )
    # test_command.run()

    # api.upload_file(
    #     path_or_fileobj="dataset/README.md",
    #     path_in_repo="README.md",
    #     repo_id="ljw20180420/CRISPR_data",
    #     repo_type="dataset"
    # )