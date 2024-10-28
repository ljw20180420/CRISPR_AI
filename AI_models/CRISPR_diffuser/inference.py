import torch
from datasets import load_dataset, Features, Value
from torch.utils.data import DataLoader
from diffusers import DiffusionPipeline
from tqdm import tqdm
from ..proxy import *
from ..config import args
from .load_data import data_collector, outputs_inference
from ..dataset.CRISPR_data import CRISPRData

CRISPR_data = CRISPRData(ref1len = args.ref1len, ref2len = args.ref2len)

@torch.no_grad()
def data_collector_inference(examples, noise_scheduler, stationary_sampler1, stationary_sampler2):
    examples2 = list()
    for example in examples:
        ref, cut = example["ref"], example["cut"]
        assert len(ref) >= args.ref1len and len(ref) >= args.ref2len, f"ref of length {len(ref)} is too short, please decrease ref1len={args.ref1len} and/or ref2len={args.ref2len} in inference arguments"
        assert cut <= args.ref1len and len(ref) - cut <= args.ref2len, f"ref1len={args.ref1len} and/or ref2len={args.ref2len} is too short, please increase them to cover cut site {cut}"
        ref1 = ref[:args.ref1len]
        ref2 = ref[-args.ref2len:]
        cut1 = cut
        cut2 = args.ref2len + cut - len(ref)
        mh_matrix, rep_num, rep_val = CRISPR_data.num2micro_homology(ref1, ref2, cut1, cut2)
        mh_matrix = CRISPR_data.get_input(ref1, ref2, cut1, cut2, mh_matrix, rep_num, rep_val, "CRISPR_diffuser")
        mh_ref2, mh_ref1 = mh_matrix.nonzero(as_tuple=True)
        examples2.append({
            'ref1': ref1,
            'ref2': ref2,
            'cut1': cut1,
            'cut2': cut2,
            'mh_ref1': mh_ref1,
            'mh_ref2': mh_ref2,
            'mh_val': mh_matrix[mh_ref2, mh_ref1],
        })
    return data_collector(examples2, noise_scheduler, stationary_sampler1, stationary_sampler2, outputs_inference)

@torch.no_grad()
def inference():
    if args.noise_scheduler == "linear":
        from .scheduler import CRISPRDiffuserLinearScheduler
        noise_scheduler = CRISPRDiffuserLinearScheduler(
            num_train_timesteps = args.noise_timesteps
        )
    elif args.noise_scheduler == "cosine":
        from .scheduler import CRISPRDiffuserCosineScheduler
        noise_scheduler = CRISPRDiffuserCosineScheduler(
            num_train_timesteps = args.noise_timesteps,
            cosine_factor = args.cosine_factor
        )
    elif args.noise_scheduler == "exp":
        from .scheduler import CRISPRDiffuserExpScheduler
        noise_scheduler = CRISPRDiffuserExpScheduler(
            num_train_timesteps = args.noise_timesteps,
            exp_scale = args.exp_scale,
            exp_base = args.exp_base
        )
    elif args.noise_scheduler == "uniform":
        from .scheduler import CRISPRDiffuserUniformScheduler
        noise_scheduler = CRISPRDiffuserUniformScheduler(
            num_train_timesteps = args.noise_timesteps,
            uniform_scale = args.uniform_scale
        )

    pipe = DiffusionPipeline.from_pretrained("ljw20180420/SX_spcas9_CRISPR_diffuser", trust_remote_code=True, custom_pipeline="ljw20180420/SX_spcas9_CRISPR_diffuser")
    pipe.unet.to(args.device)

    ds = load_dataset('json', data_files=args.inference_data, features=Features({
        'ref': Value('string'),
        'cut': Value('int16')
    }))["train"]

    inference_dataloader = DataLoader(
        dataset=ds,
        batch_size=args.batch_size,
        collate_fn=lambda examples: data_collector_inference(examples, noise_scheduler, pipe.stationary_sampler1, pipe.stationary_sampler2)
    )

    for batch in tqdm(inference_dataloader):
        yield pipe(batch, batch_size=args.batch_size, record_path=True)
