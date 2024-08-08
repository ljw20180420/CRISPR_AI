import itertools
from datasets import load_dataset, Features, Value, Sequence
from torch.utils.data import DataLoader
import os
import numpy as np
import json
import torch
import tqdm
from Predictor import gen_indel, onehotencoder, create_feature_array, features
from config import args, device

def extract_features(seq):
    assert len(seq) == 65, "sequence must be of length 65"
    guide = seq[13:33]
    indels = gen_indel(seq,30) 
    input_indel = onehotencoder(guide)
    input_ins   = onehotencoder(guide[-6:])
    input_del   = np.concatenate((create_feature_array(features, indels), input_indel), axis=None)
    return input_indel, input_ins, input_del

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

def save_lindel_dataset(ds):
    if os.path.isfile(args.data_file.parent / f"Lindel/{args.data_file.name}"):
        return
    with open(args.data_file.parent / f"Lindel/{args.data_file.name}", "w") as json_fd:
        for i in tqdm.tqdm(range(len(ds['train'])), desc="get indels and counts"):
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
                    if pos1 < record['cut1'] + 3 and pos2 > record['cut2'] - 2 and pos1 - record['cut1'] < pos2 - record['cut2']:
                        key = (pos1, pos2 - record['cut2'] + record['cut1'], "")
                        count_dict[key] = count_dict.get(key, 0) + count / corrector
                    elif pos1 >= record['cut1'] and pos2 <= record['cut2']:
                        key = (
                            record['cut1'],
                            record['cut1'],
                            record['ref1'][record['cut1']:pos1] + random_insert + record['ref2'][pos2:record['cut2']]
                        )
                        count_dict[key] = count_dict.get(key, 0) + count / corrector
            Lefts, Rights, Inserted_Seqs, counts = [], [], [], []
            for dlen in range(1, 30):
                for dstart in range(max(0, record['cut1'] - dlen - 2 + 1), record['cut1'] + 3):
                    Lefts.append(dstart)
                    Rights.append(dstart + dlen)
                    Inserted_Seqs.append("")
                    counts.append(count_dict.get((dstart, dstart + dlen, ""), 0))
            for ins in ["A", "C", "G", "T", "AA", "AC", "AG", "AT", "CA", "CC", "CG", "CT", "GA", "GC", "GG", "GT", "TA", "TC", "TG", "TT"]:
                Lefts.append(record['cut1'])
                Rights.append(record['cut1'])
                Inserted_Seqs.append(ins)
                counts.append(count_dict.get((record['cut1'], record['cut1'], ins), 0))
            Lefts.append(record['cut1'])
            Rights.append(record['cut1'])
            Inserted_Seqs.append("long")
            counts.append(sum(
                [vv for kk, vv in count_dict if len(kk[2]) > 2]
            ))
            ref = record['ref1'][:record['cut1']] + record['ref2'][record['cut2']:]
            input_indel, input_ins, input_del = extract_features(ref[record['cut1'] - 30:record['cut1'] + 35])
            fore_record = {
                'ref': ref,
                'Left': Lefts,
                'Right': Rights,
                'Inserted Seq': Inserted_Seqs,
                'count': counts,
                'input_indel': input_indel.astype(np.int8).tolist(),
                'input_ins': input_ins.astype(np.int8).tolist(),
                'input_del': input_del.astype(np.int8).tolist()
            }
            json.dump(fore_record, json_fd)
            json_fd.write('\n')

def load_lindel_dataset():
    fore_features = Features({
        'ref': Value('string'),
        'Left': Sequence(Value('int16')),
        'Right': Sequence(Value('int16')),
        'Inserted Seq': Sequence(Value('string')),
        'count': Sequence(Value('uint64')),
        'input_indel': Sequence(Value('uint8')),
        'input_ins': Sequence(Value('uint8')),
        'input_del': Sequence(Value('uint8'))
    })
    return load_dataset("json", data_files=(args.data_file.parent / f"Lindel/{args.data_file.name}").as_posix(), num_proc=12, features=fore_features)

def split_train_valid_test(ds):
    # Divide ds's train split to valid and test splits. Both has proportion test_valid_ratio.
    ds = ds['train'].train_test_split(2*args.test_valid_ratio, shuffle=True, seed=args.seed) 
    ds_valid_test = ds['test'].train_test_split(test_size=0.5, shuffle=False)
    ds['valid'] = ds_valid_test.pop('train')
    ds['test'] = ds_valid_test.pop('test')
    return ds

def collect_data(examples):
    return {
        "del_ins_count": torch.tensor([
            [sum(example['count'][:-21]), sum(example['count'][-21:])]
            for example in examples
        ], dtype=torch.uint64, device=device),
        "ins_count": torch.tensor([
            example['count'][-21:]
            for example in examples
        ], dtype=torch.uint64, device=device),
        "del_count": torch.tensor([
            example['count'][:-21]
            for example in examples
        ], dtype=torch.uint64, device=device),
        'input_indel': torch.tensor([
            example['input_indel']
            for example in examples
        ], dtype=torch.uint8, device=device),
        'input_ins': torch.tensor([
            example['input_ins']
            for example in examples
        ], dtype=torch.uint8, device=device),
        'input_del': torch.tensor([
            example['input_del']
            for example in examples
        ], dtype=torch.uint8, device=device)
    }

ds = load_align_results()
save_lindel_dataset(ds)
lindel_ds = load_lindel_dataset()
lindel_ds = split_train_valid_test(lindel_ds) 
train_dataloader = DataLoader(dataset=lindel_ds["train"], batch_size=args.batch_size, shuffle=True, collate_fn=collect_data)
valid_dataloader = DataLoader(dataset=lindel_ds["valid"], batch_size=args.batch_size, collate_fn=collect_data)
test_dataloader = DataLoader(dataset=lindel_ds["test"], batch_size=args.batch_size, collate_fn=collect_data)