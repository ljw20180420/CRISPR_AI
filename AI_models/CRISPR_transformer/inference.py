import torch
from datasets import load_dataset, Features, Value
from torch.utils.data import DataLoader
from diffusers import DiffusionPipeline
from tqdm import tqdm
from ..config import get_config, get_logger
from .load_data import data_collector, outputs_inference

args = get_config(config_file="config_CRISPR_transformer.ini")
logger = get_logger(args)

@torch.no_grad()
def data_collector_inference(examples):
    examples2 = list()
    for example in examples:
        ref, cut = example["ref"], example["cut"]
        assert len(ref) >= args.ref1len and len(ref) >= args.ref2len, f"ref of length {len(ref)} is too short, please decrease ref1len={args.ref1len} and/or ref2len={args.ref2len} in inference arguments"
        assert cut <= args.ref1len and len(ref) - cut <= args.ref2len, f"ref1len={args.ref1len} and/or ref2len={args.ref2len} is too short, please increase them to cover cut site {cut}"
        examples2.append(
            {
                "ref1": ref[:args.ref1len],
                "ref2": ref[-args.ref2len:]
            }
        )
    return data_collector(examples2, outputs_inference)

@torch.no_grad()
def inference(data_name=args.data_name, data_files="inference.json.gz"):
    logger.info("load inference data")
    ds = load_dataset('json', data_files=data_files, features=Features({
        'ref': Value('string'),
        'cut': Value('int16')
    }))["train"]

    inference_dataloader = DataLoader(
        dataset=ds,
        batch_size=args.batch_size,
        collate_fn=data_collector_inference
    )

    logger.info("setup pipeline")
    pipe = DiffusionPipeline.from_pretrained(f"{args.owner}/{data_name}_CRISPR_transformer", trust_remote_code=True, custom_pipeline=f"{args.owner}/{data_name}_CRISPR_transformer")
    pipe.CRISPR_transformer_model.to(args.device)

    for batch in tqdm(inference_dataloader):
        yield pipe(batch)
