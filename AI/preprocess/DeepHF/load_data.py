import numpy as np
import torch
import more_itertools
import Bio.SeqUtils.MeltingTemp as Tm


class SeqTokenizer:
    def __init__(self, alphabet: str) -> None:
        self.ascii_code = np.frombuffer(alphabet.encode(), dtype=np.int8)
        self.int2idx = np.empty(self.ascii_code.max() + 1, dtype=int)
        for i, c in enumerate(self.ascii_code):
            self.int2idx[c] = i

    def __call__(self, seq: str) -> np.ndarray:
        return self.int2idx[np.frombuffer(seq.encode(), dtype=np.int8)]


class TwoMerEnergy:
    def __init__(self) -> None:
        self.seq_tokenizer = SeqTokenizer("ACGT")
        self.energy = np.array(
            [
                [-0.2, -1.1, -0.9, -0.9],
                [-1.6, -2.9, -1.7, -1.8],
                [-1.5, -2.7, -2.1, -2.1],
                [-0.6, -1.3, -0.9, -1.0],
            ]
        )

    def __call__(self, seq: str) -> float:
        seq = seq.upper()
        return self.energy[
            self.seq_tokenizer[seq[:-1]],
            self.seq_tokenizer[seq[1:]],
        ].sum()


def get_energy(rnafold_sgRNA_files, rnafold_sgRNA_scaffold_files):
    (
        sgRNAs,
        stems,
        dGs,
        dG_binding_20s,
        dG_binding_7to20s,
    ) = ([], [], [], [], [])
    two_mer_energy = TwoMerEnergy()
    ext_stem = "(((((((((.((((....))))...)))))))"
    for rnafold_sgRNA_file in rnafold_sgRNA_files:
        with open(rnafold_sgRNA_file, "r") as fd:
            for _, sgRNA, second_energy in more_itertools.batched(fd, 3):
                sgRNAs.append(sgRNA)
                dGs.append(float(second_energy.split(" (")[1][:-2].strip()))
                dG_binding_20s.append(two_mer_energy(sgRNA))
                dG_binding_7to20s.append(two_mer_energy(sgRNA[7:]))

    for rnafold_sgRNA_scaffold_file in rnafold_sgRNA_scaffold_files:
        with open(rnafold_sgRNA_scaffold_file, "r") as fd:
            for _, _, second_energy in more_itertools.batched(fd, 3):
                align_seq = second_energy.split(" (")[0]
                if align_seq[18 : 18 + len(ext_stem)] == ext_stem:
                    stems.append(1.0)
                else:
                    stems.append(0.0)

    return {
        sgRNA: [stem, dG, dG_binding_20, dG_binding_7to20]
        for sgRNA, stem, dG, dG_binding_20, dG_binding_7to20 in zip(
            sgRNAs,
            stems,
            dGs,
            dG_binding_20s,
            dG_binding_7to20s,
        )
    }


@torch.no_grad()
def data_collator(
    examples: list[dict],
    ext1_up: int,
    ext1_down: int,
    ext2_up: int,
    ext2_down: int,
    energy_records: dict,
    seq_tokenizer: SeqTokenizer,
    output_observation: bool,
) -> dict:
    Xs, biological_inputs = [], []
    if output_observation:
        observations = []

    for example in examples:
        if output_observation:
            # construct observations
            observations = torch.zeros(
                (example["random_insert_uplimit"] + 2)
                * (len(example["ref2"]) + 1)
                * (len(example["ref1"]) + 1),
                dtype=torch.float64,
            )
            observations[example["ob_idx"]] = torch.tensor(
                example["ob_val"], dtype=torch.float64
            )
            # cumulate observations for all random insertion size
            observation = (
                observations.reshape(
                    example["random_insert_uplimit"] + 2,
                    len(example["ref2"]) + 1,
                    len(example["ref1"]) + 1,
                )
                .sum(axis=0)
                .flatten()
            )
            # distribute count to all positions in single micro-homology diagonal
            observation[example["mh_idx"]] = observation[example["mh_idx"]] / (
                torch.tensor(example["mh_val"]) + 1
            )
            observation = observation.reshape(
                len(example["ref2"]) + 1, len(example["ref1"]) + 1
            )
            # take the observation region based on model extension limits
            observation = observation[
                example["cut2"] - ext2_up : example["cut2"] + ext2_down,
                example["cut1"] - ext1_up : example["cut1"] + ext1_down,
            ]
            observations.append(observation)

        sgRNA21mer = example["ref1"][example["cut1"] - 17 : example["cut1"] + 4]
        Xs.append(seq_tokenizer("S" + sgRNA21mer))

        gc_count = sgRNA21mer[:20].count("G") + sgRNA21mer[:20].count("C")
        gc_above_10 = float(gc_count > 10)
        gc_below_10 = float(gc_count < 10)
        Tm_global = Tm.Tm_staluc(sgRNA21mer, rna=False)
        Tm_5mer_end = Tm.Tm_staluc(sgRNA21mer[15:21], rna=False)
        Tm_8mer_middle = Tm.Tm_staluc(sgRNA21mer[4:13], rna=False)
        Tm_4mer_start = Tm.Tm_staluc(sgRNA21mer[0:4], rna=False)

        biological_inputs.append(
            energy_records[sgRNA21mer[:20]]
            + [
                gc_above_10,
                gc_below_10,
                gc_count,
                Tm_global,
                Tm_5mer_end,
                Tm_8mer_middle,
                Tm_4mer_start,
            ]
        )

    if output_observation:
        return {
            "X": torch.from_numpy(np.stack(Xs)),
            "biological_input": torch.tensor(biological_inputs),
            "observation": torch.stack(observations),
        }
    return {
        "X": torch.from_numpy(np.stack(Xs)),
        "biological_input": torch.tensor(biological_inputs),
    }
