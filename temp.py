from datasets import load_dataset
import json
import gzip

ds = load_dataset('json', data_files="dataset/test.json.gz")["train"]

with gzip.open("test.json.gz", "wb") as gfd:
    for example in ds:
        cut1 = example["cuts"][0]["cut1"]
        cut2 = example["cuts"][0]["cut2"]
        ref1 = example["ref1"]
        ref2 = example["ref2"]
        obj = {
            "ref": ref1[:cut1] + ref2[cut2:],
            "cut": cut1
        }
        gfd.write(
            json.dumps(
                obj,
            ).encode() + b"\n"
        )
        