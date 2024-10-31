# Introduction
First of all, your should set `output_dir` in `config.ini` to a proper value based on your working environment.

If your want to train, test and upload models by yourself, please set `owner` in `config.ini` to your huggingface username. If you only do inference, left `owner` as "ljw20180420" in `config.ini`.

`data_name` is default to "SX_spcas9". To train and test on other data, either set `data_name` to one of "SX_spcas9", "SX_spymac", "SX_ispymac" in `config.ini`, or specify it by `train(data_name="SX_spcas9")` and `test(data_name="SX_spcas9")`.



# Install
```bash
git clone https://github.com/ljw20180420/CRISPR_AI.git
```



# proxy
```python
from AI_models.proxy import proxy
proxy(url="socks5h://127.0.0.1:1080")
```



# Dataset
## Upload
```python
from AI_models.dataset.upload import upload
upload()
```



# Usage
## inDelphi
### Train
```python
from AI_models.inDelphi.train import train_deletion, train_insertion
train_deletion(data_name="SX_spcas9")
train_insertion(data_name="SX_spcas9")
```
### Test and upload to huggingface
```python
from AI_models.inDelphi.test import test
test(data_name="SX_spcas9")
```
### Inference
```python
from AI_models.inDelphi.inference import inference
for output in inference(data_name="SX_spcas9", data_files="inference.json.gz"):
    pass
```
### App
```python
from AI_models.inDelphi.app import app
app(data_name="SX_spcas9")
```
### Space
```python
from AI_models.inDelphi.space import space
space(data_name="SX_spcas9")
```


## Lindel
### Train
```python
from AI_models.Lindel.train import train
train(data_name="SX_spcas9")
```
### Test and upload to huggingface
```python
from AI_models.Lindel.test import test
test(data_name="SX_spcas9")
```
### Inference
```python
from AI_models.Lindel.inference import inference
for output in inference(data_name="SX_spcas9", data_files="inference.json.gz"):
    pass
```
### App
```python
from AI_models.Lindel.app import app
app(data_name="SX_spcas9")
```
### Space
```python
from AI_models.Lindel.space import space
space(data_name="SX_spcas9")
```

## FOREcasT
### Train
```python
from AI_models.FOREcasT.train import train
train(data_name="SX_spcas9")
```
### Test and upload to huggingface
```python
from AI_models.FOREcasT.test import test
test(data_name="SX_spcas9")
```
### Inference
```python
from AI_models.FOREcasT.inference import inference
for output in inference(data_name="SX_spcas9", data_files="inference.json.gz"):
    pass
```
### App
```python
from AI_models.FOREcasT.app import app
app(data_name="SX_spcas9")
```
### Space
```python
from AI_models.FOREcasT.space import space
space(data_name="SX_spcas9")
```


## CRISPR_diffuser
### Train
```python
from AI_models.CRISPR_diffuser.train import train
train(data_name="SX_spcas9")
```
### Test and upload to huggingface
```python
from AI_models.CRISPR_diffuser.test import test
test(data_name="SX_spcas9")
```
### Inference
```python
from AI_models.CRISPR_diffuser.inference import inference
for x1ts, x2ts, ts in inference(data_name="SX_spcas9", data_files="inference.json.gz"):
    pass
```
### App
```python
from AI_models.CRISPR_diffuser.app import app
app(data_name="SX_spcas9")
```
### Space
```python
from AI_models.CRISPR_diffuser.space import space
space(data_name="SX_spcas9")
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
Add a scatter converge diagram like that in http://yang-song.net/blog/2021/score
AI explaining
UI
use latent diffuser
use mamba
use KAN
discriminative routing mechanism (https://arxiv.org/pdf/2302.05737)
sample temperature (https://arxiv.org/pdf/2302.05737)
Diffuse ref2 and ref1 independently, so that $p_\theta(x_s|x_t)=\sum_{x_0}q(x_s|x_t,x_0)p_\theta(x_0|x_t)$ https://arxiv.org/pdf/2402.03701 can be used.
```
