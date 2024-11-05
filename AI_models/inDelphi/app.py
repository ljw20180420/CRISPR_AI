import gradio as gr
from huggingface_hub import HfFileSystem
import pickle
from diffusers import DiffusionPipeline
import torch
import torch.nn.functional as F
import pandas as pd
from .inference import data_collector_inference
from ..config import get_config, get_logger

args = get_config(config_file="config_inDelphi.ini")
logger = get_logger(args)

@torch.no_grad()
def app(data_name=args.data_name):
    logger.info("setup pipeline")
    fs = HfFileSystem()
    with fs.open(f"{args.owner}/{data_name}_inDelphi/inDelphi_model/insertion_model.pkl", "rb") as fd:
        onebp_features, insert_probabilities, m654 = pickle.load(fd)
    pipe = DiffusionPipeline.from_pretrained(f"{args.owner}/{data_name}_inDelphi", trust_remote_code=True, custom_pipeline=f"{args.owner}/{data_name}_inDelphi", onebp_features = onebp_features, insert_probabilities = insert_probabilities, m654 = m654)
    pipe.inDelphi_model.to(args.device)

    def gradio_fn(ref, cut):
        batch, examples2 = data_collector_inference(
            [{
                "ref": ref,
                "cut": cut
            }],
            return_input=True
        )
        _, _, mh_del_len, _, mh_gt_pos, _ = examples2[0].values()
        mh_weight, mhless_weight, _, pre_insert_probability, pre_insert_1bp = pipe(batch).values()

        return pd.DataFrame(
            {
                "Category": ["del"] * (mh_weight[0].shape[0] + mhless_weight.shape[0]) + ["ins"] * pre_insert_1bp.shape[0],
                "Genotype position": [gt - dl for dl, gt in zip(mh_del_len, mh_gt_pos)] + [cut] * (mhless_weight.shape[0] + pre_insert_1bp.shape[0]),
                "Inserted Bases": [""] * (mh_weight[0].shape[0] + mhless_weight.shape[0]) + ["A", "C", "G", "T"],
                "Length": mh_del_len + list(range(1, mhless_weight.shape[0] + 1)) + [1] * 4,
                "Predicted frequency": (
                    F.normalize(
                        torch.cat([
                            mh_weight[0],
                            mhless_weight
                        ]),
                        p=1,
                        dim=0
                    ) * (1 - pre_insert_probability[0])
                ).tolist() + pre_insert_1bp.tolist()
            }
        )

    gr.Interface(
        fn=gradio_fn,
        inputs=["text", "number"],
        outputs=["dataframe"],
    ).launch()
