import torch
from datasets import load_dataset, Features, Value
from torch.utils.data import DataLoader
from diffusers import DiffusionPipeline
from tqdm import tqdm
from ..config import args, logger
from .load_data import data_collector, outputs_inference
from ..dataset.CRISPR_data import CRISPRData

CRISPR_data = CRISPRData(ref1len = args.ref1len, ref2len = args.ref2len, Lindel_dlen = args.Lindel_dlen, Lindel_mh_len = args.Lindel_mh_len)

@torch.no_grad()
def data_collector_inference(examples):
    examples2 = list()
    for example in examples:
        ref, cut = example["ref"], example["cut"]
        assert len(ref) >= args.ref1len and len(ref) >= args.ref2len, f"ref of length {len(ref)} is too short, please decrease ref1len={args.ref1len} and/or ref2len={args.ref2len} in inference arguments"
        assert cut <= args.ref1len and len(ref) - cut <= args.ref2len, f"ref1len={args.ref1len} and/or ref2len={args.ref2len} is too short, please increase them to cover cut site {cut}"
        assert cut >= args.Lindel_dlen - 1, f"ref upstream to cut ({cut}) is less than Lindel_dlen - 1 ({args.Lindel_dlen - 1}), extend ref to upstream"
        assert len(ref) - cut >= args.Lindel_dlen - 1, f"ref downstream to cut ({len(ref) - cut}) is less than Lindel_dlen - 1 ({args.Lindel_dlen - 1}), extend ref to downstream"
        ref1 = ref[:args.ref1len]
        ref2 = ref[-args.ref2len:]
        cut1 = cut
        cut2 = args.ref2len + cut - len(ref)
        mh_matrix, rep_num, rep_val = CRISPR_data.num2micro_homology(ref1, ref2, cut1, cut2, ext1=2, ext2=1)
        del_lens, mh_lens, dstarts, mh_idxs = CRISPR_data.get_input(ref1, ref2, cut1, cut2, mh_matrix, rep_num, rep_val, "Lindel")
        mask_del_len = (del_lens > 0).logical_and(del_lens < CRISPR_data.config.Lindel_dlen).logical_and(dstarts < 3).logical_and(dstarts + del_lens > -2)
        mask_mh_end = torch.full(mask_del_len.shape, False)
        mask_mh_end[mh_idxs] = True
        mask = mask_del_len.logical_and((mh_lens == 0).logical_or(mask_mh_end))
        mh_lens = mh_lens[mask]
        dstarts = dstarts[mask]
        dstarts[(dstarts > 0).logical_and(dstarts <= mh_lens)] = 0
        del_lens = del_lens[mask]
        mh_lens = torch.min(del_lens, mh_lens).clamp(0, CRISPR_data.config.Lindel_mh_len)
        examples2.append({
            'ref': ref,
            'cut': cut,
            'dstart': dstarts.tolist(),
            'del_len': del_lens.tolist(),
            'mh_len': mh_lens.tolist()
        })
    return data_collector(examples2, args.Lindel_dlen, args.Lindel_mh_len, outputs_inference)

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
    pipe = DiffusionPipeline.from_pretrained(f"{owner}/{data_name}_Lindel", trust_remote_code=True, custom_pipeline=f"{owner}/{data_name}_Lindel")
    pipe.indel_model.to(args.device)
    pipe.ins_model.to(args.device)
    pipe.del_model.to(args.device)

    for batch in tqdm(inference_dataloader):
        yield pipe(batch)
