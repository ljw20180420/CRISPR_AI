import gradio as gr
from huggingface_hub import HfFileSystem
import pickle
from diffusers import DiffusionPipeline
import pandas as pd
from .inference import data_collector_inference
from ..config import args, logger

def app(owner="ljw20180420", data_name="SX_spcas9"):
    logger.info("setup pipeline")
    fs = HfFileSystem()
    with fs.open(f"{owner}/{data_name}_inDelphi/inDelphi_model/insertion_model.pkl", "rb") as fd:
        onebp_features, insert_probabilities, m654 = pickle.load(fd)
    pipe = DiffusionPipeline.from_pretrained(f"{owner}/{data_name}_inDelphi", trust_remote_code=True, custom_pipeline=f"{owner}/{data_name}_inDelphi", onebp_features = onebp_features, insert_probabilities = insert_probabilities, m654 = m654)
    pipe.inDelphi_model.to(args.device)

    def gradio_fn(ref, cut):
        batch, examples2 = data_collector_inference(
            [{
                "ref": ref,
                "cut": cut
            }],
            return_input=True
        )
        _, _, mh_del_len, mh_mh_len, mh_gt_pos, _ = examples2[0].values()
        mh_weight, mhless_weight, total_del_len_weight, pre_insert_probability, pre_insert_1bp = pipe(batch).values()
        mh_output = pd.DataFrame({
            "delete_start": [gt - dl for dl, gt in zip(mh_del_len, mh_gt_pos)],
            "delete_end": mh_gt_pos,
            "micro_homology": mh_mh_len,
            "weight": mh_weight[0].tolist()
        })
        mhless_output = pd.DataFrame({
            "delete_length": list(range(1, mhless_weight.shape[0] + 1)),
            "non_mh_weight": mhless_weight.tolist(),
            "total_weight": total_del_len_weight.tolist()[0]
        })
        insert_outout = pd.DataFrame({
            "A": [pre_insert_1bp[0] * pre_insert_probability[0]],
            "C": [pre_insert_1bp[1] * pre_insert_probability[0]],
            "G": [pre_insert_1bp[2] * pre_insert_probability[0]],
            "T": [pre_insert_1bp[3] * pre_insert_probability[0]]
        })

        return mh_output, mhless_output, insert_outout

    gr.Interface(
        fn=gradio_fn,
        inputs=["text", "number"],
        outputs=["dataframe", "dataframe", "dataframe"],
    ).launch()
