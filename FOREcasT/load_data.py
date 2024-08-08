import itertools
from datasets import load_dataset, Features, Value, Sequence
from torch.utils.data import DataLoader
import subprocess
import pandas as pd
import os
import torch.nn.functional as F
import numpy as np
import json
import torch
from features import calculateFeaturesForGenIndelFile
from config import args, device
import tqdm

def load_align_results():
    def unlist_cuts(examples):
        return {
            'ref1': [ref1 for i, ref1 in enumerate(examples['ref1']) for _ in examples['cut1'][i]],
            'ref2': [ref2 for i, ref2 in enumerate(examples['ref2']) for _ in examples['cut2'][i]],
            'cut1': list(itertools.chain.from_iterable(examples['cut1'])),
            'cut2': list(itertools.chain.from_iterable(examples['cut2'])),
            'ref1_end': list(itertools.chain.from_iterable(examples['ref1_end'])),
            'ref2_start': list(itertools.chain.from_iterable(examples['ref2_start'])),
            'random_insert': list(itertools.chain.from_iterable(examples['random_insert'])),
            'count': list(itertools.chain.from_iterable(examples['count']))
        }
    alg_features = Features({
        'ref1': Value('string'),
        'ref2': Value('string'),
        'cut1': Sequence(Value('int16')),
        'cut2': Sequence(Value('int16')),
        'ref1_end': Sequence(Sequence(Value('int16'))),
        'ref2_start': Sequence(Sequence(Value('int16'))),
        'random_insert': Sequence(Sequence(Value('string'))),
        'count': Sequence(Sequence(Value('uint64')))
    })
    ds = load_dataset("json", data_files=args.data_file.as_posix(), num_proc=12, features=alg_features)
    return ds.map(unlist_cuts, batched=True, desc="unlist cuts in dataset")

def save_tmp_genindels_features_file(ds):
    os.makedirs(args.data_file.parent / "FOREcasT/tmp", exist_ok=True)
    for i in tqdm.tqdm(range(len(ds['train'])), desc="get temp file of genindels and features"):
        tmp_genindels_file = (args.data_file.parent / f"FOREcasT/tmp/tmp_genindels_{i}.txt").as_posix()
        tmp_genindels_file_next = (args.data_file.parent / f"FOREcasT/tmp/tmp_genindels_{i + 1}.txt").as_posix()
        tmp_features_file = (args.data_file.parent / f"FOREcasT/tmp/tmp_features_{i}.txt").as_posix()
        if not os.path.isfile(tmp_features_file):
            # merge ref1 and ref1 to ref
            ref = ds['train'][i]['ref1'][:ds['train'][i]['cut1']] + ds['train'][i]['ref2'][ds['train'][i]['cut2']:]
            subprocess.run(["FOREcasT/indelmap/indelgentarget", ref, f"{ds['train'][i]['cut1'] + 3}", tmp_genindels_file])
        if not os.path.isfile(tmp_genindels_file_next):
            calculateFeaturesForGenIndelFile(tmp_genindels_file, ds['train'][i]['ref1'], ds['train'][i]['cut1'], tmp_features_file)

def save_fore_dataset(ds):
    if os.path.isfile(args.data_file.parent / f"FOREcasT/{args.data_file.name}"):
        return
    with open(args.data_file.parent / f"FOREcasT/{args.data_file.name}", "w") as json_fd:
        for i in tqdm.tqdm(range(len(ds['train'])), desc="get indels and counts"):
            tmp_feature_file = (args.data_file.parent / f"FOREcasT/tmp/tmp_features_{i}.txt").as_posix()
            record = ds['train'][i]
            count_dict = {}
            for ref1_end, ref2_start, random_insert, count in zip(record['ref1_end'], record['ref2_start'], record['random_insert'], record['count']):
                # expand MH to all possible locations
                up1, up2 = ref1_end, ref2_start
                while up1 < record['cut1'] and up2 < len(record['ref2']):
                    if not random_insert and record['ref1'][up1] == record['ref2'][up2]:
                        up1 += 1
                        up2 += 1
                    else:
                        break
                down1, down2 = ref1_end, ref2_start
                while down1 > 0 and down2 > record['cut2']:
                    if not random_insert and record['ref1'][down1 - 1] == record['ref2'][down2 - 1]:
                        down1 -= 1
                        down2 -= 1
                    else:
                        break
                corrector = 1 if args.not_correct_micro_homology else (up1 - down1 + 1)
                for pos1, pos2 in zip(range(down1, up1 + 1), range(down2, up2 + 1)):
                    if pos1 <= record['cut1'] and pos2 >= record['cut2'] and pos1 - record['cut1'] < pos2 - record['cut2']:
                        key = (pos1, pos2 - record['cut2'] + record['cut1'], "")
                        count_dict[key] = count_dict.get(key, 0) + count / corrector
                    elif pos1 >= record['cut1'] and pos2 <= record['cut2']:
                        key = (
                            record['cut1'],
                            record['cut1'],
                            record['ref1'][record['cut1']:pos1] + random_insert + record['ref2'][pos2:record['cut2']]
                        )
                        count_dict[key] = count_dict.get(key, 0) + count / corrector
            feature_data = pd.read_csv(tmp_feature_file, skiprows=2, sep='\t', dtype={'Inserted Seq':str})
            feature_data['Left'] += 1
            feature_data['Inserted Seq'] = feature_data['Inserted Seq'].fillna("")
            data = feature_data.drop(columns=['Indel', 'Left', 'Right', 'Inserted Seq']).values
            rows, cols = np.nonzero(data)
            fore_record = {
                'ref': record['ref1'][:record['cut1']] + record['ref2'][record['cut2']:],
                'Left': feature_data['Left'].values.tolist(),
                'Right': feature_data['Right'].values.tolist(),
                'Inserted Seq': feature_data['Inserted Seq'].fillna("").values.tolist(),
                'count': [count_dict.get((row['Left'], row['Right'], row['Inserted Seq']), 0) for _, row in feature_data.iterrows()],
                'feature_num': data.shape[1],
                'row': rows.tolist(),
                'col': cols.tolist()
            }
            json.dump(fore_record, json_fd)
            json_fd.write('\n')

def load_fore_dataset():
    fore_features = Features({
        'ref': Value('string'),
        'Left': Sequence(Value('int16')),
        'Right': Sequence(Value('int16')),
        'Inserted Seq': Sequence(Value('string')),
        'count': Sequence(Value('float32')),
        'feature_num': Value('uint32'),
        'row': Sequence(Value('uint32')),
        'col': Sequence(Value('uint32'))
    })
    return load_dataset("json", data_files=(args.data_file.parent / f"FOREcasT/{args.data_file.name}").as_posix(), num_proc=12, features=fore_features)

def split_train_valid_test(ds):
    # Divide ds's train split to valid and test splits. Both has proportion test_valid_ratio.
    ds = ds['train'].train_test_split(2*args.test_valid_ratio, shuffle=True, seed=args.seed) 
    ds_valid_test = ds['test'].train_test_split(test_size=0.5, shuffle=False)
    ds['valid'] = ds_valid_test.pop('train')
    ds['test'] = ds_valid_test.pop('test')
    return ds

def collect_data(examples):
    return {
        "count": torch.tensor([
            example['count']
            for example in examples
        ], dtype=torch.float32, device=device),
        "data": torch.stack([
            torch.sparse_coo_tensor(
                [example['row'], example['col']], torch.ones(len(example['row'])), (len(example['count']), example['feature_num']),
                dtype=torch.int8,
                device=device
            ).to_dense()
            for example in examples
        ])
    }

ds = load_align_results()
save_tmp_genindels_features_file(ds)
save_fore_dataset(ds)
fore_ds = load_fore_dataset()
fore_ds = split_train_valid_test(fore_ds) 
train_dataloader = DataLoader(dataset=fore_ds["train"], batch_size=args.batch_size, shuffle=True, collate_fn=collect_data)
valid_dataloader = DataLoader(dataset=fore_ds["valid"], batch_size=args.batch_size, collate_fn=collect_data)
test_dataloader = DataLoader(dataset=fore_ds["test"], batch_size=args.batch_size, collate_fn=collect_data)