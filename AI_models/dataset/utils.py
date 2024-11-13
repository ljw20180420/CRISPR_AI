from huggingface_hub import HfApi
from pathlib import Path
from datasets.commands.test import TestCommand
from ..config import get_config

args = get_config()

def test():
    test_command = TestCommand(
        dataset="AI_models/dataset/CRISPR_data.py",
        name=None,
        cache_dir=None,
        data_dir=None,
        all_configs=True,
        save_infos=True,
        ignore_verifications=False,
        force_redownload=False,
        clear_cache=False,
        num_proc=None,
        trust_remote_code=True
    )
    test_command.run()

def upload(do_test=False):
    api = HfApi()
    api.create_repo(
        repo_id=f"{args.owner}/CRISPR_data",
        repo_type="dataset",
        exist_ok=True
    )
    api.upload_file(
        path_or_fileobj=Path(__file__).parent / "CRISPR_data.py",
        path_in_repo="CRISPR_data.py",
        repo_id=f"{args.owner}/CRISPR_data",
        repo_type="dataset"
    )

    if do_test:
        test()
        api.upload_file(
            path_or_fileobj="AI_models/dataset/README.md",
            path_in_repo="README.md",
            repo_id=f"{args.owner}/CRISPR_data",
            repo_type="dataset"
        )
    