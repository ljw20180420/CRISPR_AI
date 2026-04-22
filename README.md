# Introduction

This repository includes several models to predict the CRISPR editing products from the sgRNA sequences.

# Install

This repo use conda to manage packages.
```shell
$ git clone https://github.com/ljw20180420/CRISPR_AI.git
$ cd CRISPR_AI
$ conda create -p ./.conda
$ conda env update -p ./.conda --file environment.yml
```
For old pascal gpu, after the above steps,
```shell
$ pip install -U -r requirements_torch_pascal.txt
```

# Usage

To list all subcommands,
```shell
$ ./run.py --help
```

## Inference

```shell
$ ./run.py infer --help
```
Example,
```shell
$ ./run.py infer \
  --input AI/dataset/inference.csv.gz \
  --output inference_output.csv \
  --inference AI.inference.MyInference \
  --inference.ext1_up 25 \
  --inference.ext1_down 6 \
  --inference.ext2_up 6 \
  --inference.ext2_down 25 \
  --inference.max_del_size 0 \
  --test.checkpoints_path ljw20180420/CRIfuser_CRIfuser_SX_ispymac \
  --test.logs_path ljw20180420/CRIfuser_CRIfuser_SX_ispymac \
  --test.target GreatestCommonCrossEntropy \
  --test.maximize_target false \
  --test.overwrite {}
```
Set both `--test.checkpoints_path` and `--test.logs_path` to the huggingface repository. Set `--inference.max_del_size 44` for `inDelphi`, `FOREcasT`, `Lindel`. Set `--inference.max_del_size 0` for other models.

The gradio app is available for `CRIfuser` trained on `spycas9`, `spymac`, `ispymac`.
```
$ scripts/formal/app.sh
```
The graido app is also available on the huggingface space at https://huggingface.co/spaces/ljw20180420/CRISPR_AI


# Train and test

For helps, execute
```console
./run.py -h
./run.py train -h
./run.py test -h
```
The example training config is `AI/preprocess/train.yaml`. The example testing config is `AI/preprocess/test.yaml`. The example model configs is `AI/preprocess/[preprocess]/[model_type].yaml`. `defaults.sh` containes example runs.

# TODO

- wait for the resolve of the conflicts between `gr.Dataframe` and `gr.Radio`/`gr.Dropdown`

- 训练一个额外的xgboost作为随机插入模型，和CRIfuser独立

- CRIfuser损失函数bench
- 用__all__来去除不必要的.preprocess和.model
- 把AI重命名成CRISPR_AI，把run.py移动到CRISPR_AI下的__main__.py
- shap.py现在支持读取pandas（pull requests），因此不需要再转化为numpy了

- improve CRIfuser inference
  - define an importance weight according to step and distance from the cleavage site
  - distribute the sampled outcomes according to the importance weights
  - redo shap for CRIfuser (long term)
    - either for sample step 1
    - or use new inference method
    - in either case, modify the inference method for shap in paper
- Use hf pipeline.
- add space time compare of models
- Add benchmark for accuracy of CRIfuser with different downsampling
- Figures
  - profile heatmap
  - diffusion dynamics
  - diagram, unet two trapezoid with layers, input sequence one-hot small blocks, microhomology blocks, model header circle lines.
  - Convolution kernel heatmap.
  - Metrics hexagon.
  - metric distribution.
  - frame shift.
- More metrics.
- Train CRIfuser model for other metrics.
- Read SHAP and huggingface hub.
- shap
  - various plot.
  - interactions
  - deeplift visualization
  - shap.multioutput_decision_plot to compare multiple models
  - shap.plots.group_difference to compare multiple models
  - shap.plots.bar with groups to compare multiple models
  - shap.plots.embedding to compare multiple models (supervised clustering of models)
  - merge shap values according to biomarks like mmej
- Benchmarks.
  - Draw epoch-status during training process, including XGBoost.
- hyperparameter choice (optuna) for models with good benchmarks.
- xgboost interpret.
  - get_score
  - pred_contribs
  - pred_interactions
  - plot_importance
  - plot_tree
- LightGBM, XGBoost, CatBoost, hmmlearn, seqlearn, pystruct, InterpretML
- Use model.eval() in test and inference
- Add api documentation after finish the project
- train use float32

- noise2noise explanation of prediction results
- put more weights on small steps of diffusion
- increase model parameters
- use latent diffuser
- use mamba
- use KAN
- discriminative routing mechanism (https://arxiv.org/pdf/2302.05737)
- sample temperature (https://arxiv.org/pdf/2302.05737)
