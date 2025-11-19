from typing import Literal
import torch
import pandas as pd
import gradio as gr
from common_ai.gradio_fn import MyGradioFnAbstract


class MyGradioFn(MyGradioFnAbstract):
    @torch.no_grad()
    def __call__(
        self, repo_id: str, ref: str, cut: int, scaffold: Literal["spcas9", "spymac"]
    ) -> pd.DataFrame:
        self.reload_inference(repo_id=repo_id)
        infer_df = pd.DataFrame(
            {
                "ref": [ref],
                "cut": [cut],
                "scaffold": [scaffold],
            }
        )

        return self.my_inference(infer_df, self.test_cfg, self.train_parser)

    def launch(self):
        gr.Interface(
            fn=self,
            inputs=[
                gr.Dropdown(choices=self.inference_dict.keys()),
                "text",
                "number",
                gr.Dropdown(choices=["spcas9", "spymac"]),
            ],
            outputs=["dataframe"],
        ).launch()
