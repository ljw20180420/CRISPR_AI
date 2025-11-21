from typing import Literal
import torch
import pandas as pd
import gradio as gr
from common_ai.gradio_fn import MyGradioFnAbstract
import hashlib
import tempfile


class MyGradioFn(MyGradioFnAbstract):
    @torch.no_grad()
    def __call__(
        self, repo_id: str, ref: str, cut: int, scaffold: Literal["spcas9", "spymac"]
    ) -> tuple[pd.DataFrame, str]:
        self.reload_inference(repo_id=repo_id)
        infer_df = pd.DataFrame(
            {
                "ref": [ref],
                "cut": [cut],
                "scaffold": [scaffold],
            }
        )
        infer_out = self.my_inference(infer_df, self.test_cfg, self.train_parser)
        infer_out_hash = hashlib.md5(infer_out.to_string().encode()).hexdigest()
        temp_dir = self.DEFAULT_TEMP_DIR / infer_out_hash
        temp_dir.mkdir(exist_ok=True, parents=True)
        infer_out_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir).name
        infer_out.to_csv(infer_out_file, index=False)

        return infer_out, infer_out_file

    def launch(self):
        gr.Interface(
            fn=self,
            inputs=[
                gr.Dropdown(choices=self.inference_dict.keys()),
                "text",
                "number",
                gr.Dropdown(choices=["spcas9", "spymac"]),
            ],
            outputs=["dataframe", "file"],
        ).launch()
