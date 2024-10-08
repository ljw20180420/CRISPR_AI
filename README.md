

discriminative routing mechanism (https://arxiv.org/pdf/2302.05737)

sample temperature (https://arxiv.org/pdf/2302.05737)

Diffuse ref2 and ref1 independently, so that $p_\theta(x_s|x_t)=\sum_{x_0}q(x_s|x_t,x_0)p_\theta(x_0|x_t)$ https://arxiv.org/pdf/2402.03701 can be used.

improve sample method for DataLoader

make an improvement version for continuous time (tandom? diagnal?)

Predictor-Corrector

Direct Denoising Model Supervision

Continuous Time ELBO with Factorization

partial time steps

Unet learn distribution instead of point value

migrate from argparse to ConfigArgParse

use latent diffuser

use unet from timm

use mamba

use TCN

use transformer-TCN

use transformer-encoder-decoder to predict DNA

predict from both sides to the middle

add figures to tensorboard

use torch.nn.functional.normalize to normalize observations

TODO
```list
Give up transformers pipeline. Always use diffusers pipeline because it is more flexible.
Datasets 3.0.0 has bug with hf-mirror.com. Load data on huggingface.co through proxy.
Put proxy setting in a single file.
Put all AI models in a single folder (thereby being a single package)
Remove N from model (only ACGT) for performance
Add a scatter converge diagram like that in http://yang-song.net/blog/2021/score
AI explaining
Tensorboard
upload huggingface
UI
```
