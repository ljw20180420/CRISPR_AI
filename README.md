# Introduction

This repository includes several models to predict the CRISPR editing products from the sgRNA sequences. The dataset is held on anthor repository [here](https://github.com/ljw20180420/CRISPRdata).



# Install

```console
$ git clone https://github.com/ljw20180420/CRISPR_AI.git
$ cd CRISPR_AI
$ conda install --file requirements_conda.txt
$ pip install -r requirements_pip.txt
```
For old pascal gpu,
```console
$ git clone https://github.com/ljw20180420/CRISPR_AI.git
$ cd CRISPR_AI
$ conda install --file requirements_conda.txt
$ pip install -r requirements_torch_pascal.txt
$ pip install -r requirements_pip.txt
```
If you have problem with gradio, upgrade it.
```console
$ pip install --upgrade gradio
```

# Train and test

For helps, execute
```console
./run.py -h
./run.py train -h
./run.py test -h
```
The example training config is `AI/preprocess/train.yaml`. The example testing config is `AI/preprocess/test.yaml`. The example model configs is `AI/preprocess/[preprocess]/[model_type].yaml`. `defaults.sh` containes example runs.

# Hyperparameter search

The hyperparameter-searching is based on [Optuna](https://optuna.readthedocs.io). For helps. execute
```console
./hyperparameters.py -h
```

# Benchmark

Train and test models with default parameters by
```
./defaults.sh
```
Compare loss functions for `CRIfuser` by
```
./CRIfuser_loss_functions.sh
```
Summarize benchmarks results by
```bash
./benchmark.py
```

# TODO

- early stopping (https://github.com/Bjarten/early-stopping-pytorch). This is inspired by the ignite EarlyStopping class.
- LightGBM, XGBoost, hmmlearn
- upload, inference, app, space
- model interpretable
- Use model.eval() in test and inference
- Use os.PathLike
- do pre-computation when initialize data collator and model so that subsequent computations are faster
- hyperparameter choice (optuna)
- Add api documentation after finish the project
- train use float32
- add type hint

- parallel data_collator, especially for FOREcasT

- noise2noise explanation of prediction results
- put more weights on small steps of diffusion
- increase model parameters
- decrease model window size (128 -> 64 -> 32)
- use latent diffuser
- use mamba
- use KAN
- discriminative routing mechanism (https://arxiv.org/pdf/2302.05737)
- sample temperature (https://arxiv.org/pdf/2302.05737)
