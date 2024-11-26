# Introduction
First of all, your should set `output_dir` in `config.ini` to a proper value based on your working environment.

If your want to train, test and upload models by yourself, please set `owner` in `config.ini` to your huggingface username. If you only do inference, left `owner` as "ljw20180420" in `config.ini`.

`data_name` is default to "SX_spcas9". To train and test on other data, either set `data_name` to one of "SX_spcas9", "SX_spymac", "SX_ispymac" in `config.ini`, or specify it by `train(data_name="SX_spcas9")` and `test(data_name="SX_spcas9")`.



# Install
```bash
git clone https://github.com/ljw20180420/CRISPR_AI.git
```



# Original inDelphi python environment
```bash
conda env create --prefix AI_models/inDelphi/reference/.conda --file AI_models/inDelphi/reference/inDelphi.yaml
```



# Login
```python
from huggingface_hub import login
login(add_to_git_credential=True, new_session=False, write_permission=True)
```



# Proxy
## huggingface_hub
```python
from AI_models.proxy import proxy
proxy(url="socks5h://127.0.0.1:1080")
```
## datasets
Add the following to `.bashrc`.
```bash
export MY_HF_DATASETS_DOWNLOAD_MANAGER_PROXY="socks5h://127.0.0.1:1080"
```



# Dataset
## Test
```python
from AI_models.dataset.utils import test
test()
```


## Upload
```python
from AI_models.dataset.utils import upload
upload(do_test=False)
upload(do_test=True)
```



# Benchmark
```bash
./benchmark.py
```



# Usage
## inDelphi
### Train
```python
from AI_models.inDelphi.train import train_deletion, train_insertion
train_deletion(data_name="SX_spcas9") # SX_spymac, SX_ispymac
train_insertion(data_name="SX_spcas9")
```
### Test and upload to huggingface
```python
from AI_models.inDelphi.test import test
test(data_name="SX_spcas9") # SX_spymac, SX_ispymac
```
### Inference
```python
from AI_models.inDelphi.inference import inference
for output in inference(data_name="SX_spcas9", data_files="inference.json.gz"): # SX_spymac, SX_ispymac
    pass
```
### App
```python
from AI_models.inDelphi.app import app
app(data_name="SX_spcas9") # SX_spymac, SX_ispymac
```
### Space
```python
from AI_models.inDelphi.space import space
space(data_name="SX_spcas9") # SX_spymac, SX_ispymac
```


## Lindel
### Train
```python
from AI_models.Lindel.train import train
train(data_name="SX_spcas9") # SX_spymac, SX_ispymac
```
### Test and upload to huggingface
```python
from AI_models.Lindel.test import test
test(data_name="SX_spcas9") # SX_spymac, SX_ispymac
```
### Inference
```python
from AI_models.Lindel.inference import inference
for output in inference(data_name="SX_spcas9", data_files="inference.json.gz"): # SX_spymac, SX_ispymac
    pass
```
### App
```python
from AI_models.Lindel.app import app
app(data_name="SX_spcas9") # SX_spymac, SX_ispymac
```
### Space
```python
from AI_models.Lindel.space import space
space(data_name="SX_spcas9") # SX_spymac, SX_ispymac
```

## FOREcasT
### Train
```python
from AI_models.FOREcasT.train import train
train(data_name="SX_spcas9") # SX_spymac, SX_ispymac
```
### Test and upload to huggingface
```python
from AI_models.FOREcasT.test import test
test(data_name="SX_spcas9") # SX_spymac, SX_ispymac
```
### Inference
```python
from AI_models.FOREcasT.inference import inference
for output in inference(data_name="SX_spcas9", data_files="inference.json.gz"): # SX_spymac, SX_ispymac
    pass
```
### App
```python
from AI_models.FOREcasT.app import app
app(data_name="SX_spcas9") # SX_spymac, SX_ispymac
```
### Space
```python
from AI_models.FOREcasT.space import space
space(data_name="SX_spcas9") # SX_spymac, SX_ispymac
```


## CRISPR diffuser
### Train
```python
from AI_models.CRISPR_diffuser.train import train
train(data_name="SX_spcas9") # SX_spymac, SX_ispymac
```
### Test and upload to huggingface
```python
from AI_models.CRISPR_diffuser.test import test
test(data_name="SX_spcas9") # SX_spymac, SX_ispymac
```
### Inference
```python
from AI_models.CRISPR_diffuser.inference import inference
for x1ts, x2ts, ts in inference(data_name="SX_spcas9", data_files="inference.json.gz"): # SX_spymac, SX_ispymac
    pass
```
### App
```python
from AI_models.CRISPR_diffuser.app import app
app(data_name="SX_spcas9") # SX_spymac, SX_ispymac
```
### Space
```python
from AI_models.CRISPR_diffuser.space import space
space(data_name="SX_spcas9") # SX_spymac, SX_ispymac
```
### Dynamics
```python
from AI_models.CRISPR_diffuser.dynamics import dynamics
dynamics()
from AI_models.CRISPR_diffuser.dynamics import call_forward_reverse
call_forward_reverse()
```


## CRISPR transformer
### Train
```python
from AI_models.CRISPR_transformer.train import train
train(data_name="SX_spcas9") # SX_spymac, SX_ispymac
```
### Test and upload to huggingface
```python
from AI_models.CRISPR_transformer.test import test
test(data_name="SX_spcas9") # SX_spymac, SX_ispymac
```
### Inference
```python
from AI_models.CRISPR_transformer.inference import inference
for output in inference(data_name="SX_spcas9", data_files="inference.json.gz"): # SX_spymac, SX_ispymac
    pass
```
### App
```python
from AI_models.CRISPR_transformer.app import app
app(data_name="SX_spcas9") # SX_spymac, SX_ispymac
```
### Space
```python
from AI_models.CRISPR_transformer.space import space
space(data_name="SX_spcas9") # SX_spymac, SX_ispymac
```



# Input
```json
{"ref": "AAAAAAAAAAAAAAAAAAAAAAAAGACGGCAGCCTTTTGACCTCCCAACCCCCCTATAGTCAGATAGTCAAGAAGGGCATTATCTGGCTTACCTGAATCGTCCCAAGAATTTTCTTCGGTGAGCATTTGTGGAGACCCTGGGATGTAGGTTGGATTAAACTGTGATGGGTCCATCGGCGTCTTGACACAACACTAGGCTT", "cut": 100}
{"ref": "AAAAAAAAAAAAAAAAAAAAAAAAGACGGCAGCCTTTTGACCTCCCAACCCCCCTATAGTCAGATAGTCAAGAAGGGCATTATCTGGCTTACCTGAATCGTCCCGGGAATTTTCTTCGGTGAGCATTTGTGGAGACCCTGGGATGTAGGTTGGATTAAACTGTGATGGGTCCATCGGCGTCTTGACACAACACTAGGCTT", "cut": 100}
{"ref": "AAAAAAAAAAAAAAAAAAAAGACCCTTCAGTGCTAAGGCACCTCTAATGCTCTCTTCATTGACCTTATCCCGTTTAACTCCTCAGATGAACGCCTCACAGCTGAAAAGATGGATGAGCAGAGGCGGCAGAATGTTGCCTATCAGTACCTGTGCCGGCTGGAGGAGGCCAAGCGGTGAGCGGAGTCCAGGAAGATGGACTC", "cut": 100}
{"ref": "AAAAAAAAAAAAAAAAAAAAGACCCTTCAGTGCTAAGGCACCTCTAATGCTCTCTTCATTGACCTTATCCCGTTTAACTCCTCAGATGAACGCCTCACAGCTGAGGAGATGGATGAGCAGAGGCGGCAGAATGTTGCCTATCAGTACCTGTGCCGGCTGGAGGAGGCCAAGCGGTGAGCGGAGTCCAGGAAGATGGACTC", "cut": 100}
{"ref": "AAAAAAAAAAAAAAAAGGGACACACAGACTTCAAGTTTCAAAATAAAATGTGAAATTCATTAGCTCTGAAAACAATACTTACAACTGAAATGAACACATTTGTAAAATCTAATAATTCTGTCCATTGAAGAAATCGTCGAATAAAGGACTTAGGAGGGAGAAAAGCAACAGAGAGGTTAATGGCAGCAGAAAAATAAAGA", "cut": 100}
{"ref": "AAAAAAAAAAAAAAAAGGGACACACAGACTTCAAGTTTCAAAATAAAATGTGAAATTCATTAGCTCTGAAAACAATACTTACAACTGAAATGAACACATTTGTAGGATCTAATAATTCTGTCCATTGAAGAAATCGTCGAATAAAGGACTTAGGAGGGAGAAAAGCAACAGAGAGGTTAATGGCAGCAGAAAAATAAAGA", "cut": 100}
{"ref": "AAAAAAAAAAAAAAAATGAAAATAAAAATCCTAATATGAAGGAGTGTGGGTGAAGCGTGGATCACTGTTACTGAATCACCCTGCAGATGCTGTCATCCTATTGTAAAAGGTGAATGATAACAAGGAGCCGGAGCAGATCGCTTTTCAGGATGAGGACGAGGCCCAGCTGAACAAGGAGAACTGGACGGTTGTGAAGACTC", "cut": 100}
{"ref": "AAAAAAAAAAAAAAAATGAAAATAAAAATCCTAATATGAAGGAGTGTGGGTGAAGCGTGGATCACTGTTACTGAATCACCCTGCAGATGCTGTCATCCTATTGTGGAAGGTGAATGATAACAAGGAGCCGGAGCAGATCGCTTTTCAGGATGAGGACGAGGCCCAGCTGAACAAGGAGAACTGGACGGTTGTGAAGACTC", "cut": 100}
{"ref": "AAAAAAAAAAAAACAGTGAAAAGCAATCCCCTTACCACACATGCTCCAACCCCACCCCTCCCACCCTGCTGCCCCCATGTACACTTACACATTAGTGTGAAGCTAAAATTCATCAGTCTTGTAGCCAACTGCAAAGTTGCTCTGGGTCACTCGGGATTTTGCAGTCTCAAAATTCATCTGGTAGCCGGCCAGCCAGCCCT", "cut": 100}
{"ref": "AAAAAAAAAAAAACAGTGAAAAGCAATCCCCTTACCACACATGCTCCAACCCCACCCCTCCCACCCTGCTGCCCCCATGTACACTTACACATTAGTGTGAAGCTGGAATTCATCAGTCTTGTAGCCAACTGCAAAGTTGCTCTGGGTCACTCGGGATTTTGCAGTCTCAAAATTCATCTGGTAGCCGGCCAGCCAGCCCT", "cut": 100}
```





TODO
```list
more robust valid loss for CRISPR diffusion
put more weights on small steps
increase model parameters
decrease model window size (128 -> 64 -> 32)
for decreased window size, exact inference is possible
AI explaining
use latent diffuser
use mamba
use KAN
discriminative routing mechanism (https://arxiv.org/pdf/2302.05737)
sample temperature (https://arxiv.org/pdf/2302.05737)
Diffuse ref2 and ref1 independently, so that $p_\theta(x_s|x_t)=\sum_{x_0}q(x_s|x_t,x_0)p_\theta(x_0|x_t)$ https://arxiv.org/pdf/2402.03701 can be used.
```
