import torch
from datasets import load_dataset, Features, Value
from torch.utils.data import DataLoader
from diffusers import DiffusionPipeline
from tqdm import tqdm
from ..proxy import *
from ..config import args
from .load_data import data_collector
from ..dataset.CRISPR_data import CRISPRData

CRISPR_data = CRISPRData(ref1len = args.ref1len, ref2len = args.ref2len, FOREcasT_MAX_DEL_SIZE = args.FOREcasT_MAX_DEL_SIZE)

@torch.no_grad()
def data_collector_inference(examples):
    for example in examples:
        ref, cut = example["ref"], example["cut"]
        assert len(ref) >= args.ref1len and len(ref) >= args.ref2len, f"ref of length {len(ref)} is too short, please decrease ref1len={args.ref1len} and/or ref2len={args.ref2len} in inference arguments"
        assert cut <= args.ref1len and len(ref) - cut <= args.ref2len, f"ref1len={args.ref1len} and/or ref2len={arg.ref2len} is too short, please increase them to cover cut site {cut}"
        assert cut >= args.FOREcasT_MAX_DEL_SIZE, f"ref upstream to cut ({cut}) is less than FOREcasT_MAX_DEL_SIZE ({args.FOREcasT_MAX_DEL_SIZE}), extend ref to upstream"
        assert len(ref) - cut >= args.FOREcasT_MAX_DEL_SIZE, f"ref downstream to cut ({len(ref) - cut}) is less than FOREcasT_MAX_DEL_SIZE ({args.FOREcasT_MAX_DEL_SIZE}), extend ref to downstream"
    return data_collector(examples, output_count=False)

@torch.no_grad()
def inference():
    ds = load_dataset('json', data_files=args.inference_data, features=Features({
        'ref': Value('string'),
        'cut': Value('int16')
    }))["train"]

    inference_dataloader = DataLoader(
        dataset=ds,
        batch_size=args.batch_size,
        collate_fn=data_collector_inference
    )

    pipe = DiffusionPipeline.from_pretrained("ljw20180420/SX_spcas9_FOREcasT", trust_remote_code=True, custom_pipeline="ljw20180420/SX_spcas9_FOREcasT", MAX_DEL_SIZE=args.FOREcasT_MAX_DEL_SIZE)
    pipe.FOREcasT_model.to(args.device)

    for batch in tqdm(inference_dataloader):
        yield pipe(batch)
