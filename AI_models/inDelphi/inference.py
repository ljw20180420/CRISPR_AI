import torch
from datasets import load_dataset, Features, Value
from torch.utils.data import DataLoader
from diffusers import DiffusionPipeline
from tqdm import tqdm
from huggingface_hub import HfFileSystem
import pickle
from ..config import args, logger
from .load_data import data_collector, outputs_inference
from ..dataset.CRISPR_data import gc_content, CRISPRData

CRISPR_data = CRISPRData(ref1len = args.ref1len, ref2len = args.ref2len, DELLEN_LIMIT = args.DELLEN_LIMIT)

@torch.no_grad()
def data_collector_inference(examples):
    examples2 = list()
    for example in examples:
        ref, cut = example["ref"], example["cut"]
        assert len(ref) >= args.ref1len and len(ref) >= args.ref2len, f"ref of length {len(ref)} is too short, please decrease ref1len={args.ref1len} and/or ref2len={args.ref2len} in inference arguments"
        assert cut <= args.ref1len and len(ref) - cut <= args.ref2len, f"ref1len={args.ref1len} and/or ref2len={args.ref2len} is too short, please increase them to cover cut site {cut}"
        if cut < args.DELLEN_LIMIT - 1:
            logger.warning(f"ref length upstream to cut ({cut}) is less than DELLEN_LIMIT - 1 ({args.DELLEN_LIMIT - 1}), no micro-homology will be checked beyond there")
        if len(ref) - cut < args.DELLEN_LIMIT - 1:
            logger.warning(f"ref length downstream to cut ({len(ref) - cut}) is less than DELLEN_LIMIT - 1 ({args.DELLEN_LIMIT - 1}), no micro-homology will be checked beyond there")
        ref1 = ref[:args.ref1len]
        ref2 = ref[-args.ref2len:]
        cut1 = cut
        cut2 = args.ref2len + cut - len(ref)
        mh_matrix, rep_num, rep_val = CRISPR_data.num2micro_homology(ref1, ref2, cut1, cut2)
        del_lens, mh_lens, gt_poss = CRISPR_data.get_input(ref1, ref2, cut1, cut2, mh_matrix, rep_num, rep_val, "inDelphi")
        mask_del = (del_lens > 0) & (del_lens < CRISPR_data.config.DELLEN_LIMIT) & (gt_poss >= cut1) & (gt_poss - del_lens <= cut1)
        del_lens, mh_lens, gt_poss = del_lens[mask_del], mh_lens[mask_del], gt_poss[mask_del]
        mask_mh = mh_lens > 0
        examples2.append(
            {
                "ref": ref,
                "cut": cut,
                "mh_del_len": del_lens[mask_mh].tolist(),
                "mh_mh_len": mh_lens[mask_mh].tolist(),
                "mh_gc_frac": [gc_content(ref[gt_pos - mh_len:gt_pos]) for mh_len, gt_pos in zip(mh_lens[mask_mh], gt_poss[mask_mh])]
            }
        )
    return data_collector(examples2, args.DELLEN_LIMIT, outputs_inference)

@torch.no_grad()
def inference(owner="ljw20180420", data_name="SX_spcas9", data_files="inference.json.gz"):
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
    fs = HfFileSystem()
    with fs.open(f"{owner}/{data_name}_inDelphi/inDelphi_model/insertion_model.pkl", "rb") as fd:
        onebp_features, insert_probabilities, m654 = pickle.load(fd)
    pipe = DiffusionPipeline.from_pretrained(f"{owner}/{data_name}_inDelphi", trust_remote_code=True, custom_pipeline=f"{owner}/{data_name}_inDelphi", onebp_features = onebp_features, insert_probabilities = insert_probabilities, m654 = m654)
    pipe.inDelphi_model.to(args.device)

    for batch in tqdm(inference_dataloader):
        yield pipe(batch)