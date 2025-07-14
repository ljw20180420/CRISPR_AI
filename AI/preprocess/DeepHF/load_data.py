import numpy as np
import torch
import more_itertools
import Bio.SeqUtils.MeltingTemp as Tm
import subprocess
from ..utils import GetMH, SeqTokenizer


class TwoMerEnergy:
    def __init__(self) -> None:
        self.seq_tokenizer = SeqTokenizer("ACGU")
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
            self.seq_tokenizer(seq[:-1]),
            self.seq_tokenizer(seq[1:]),
        ].sum()


class DataCollator:
    def __init__(
        self,
        ext1_up: int,
        ext1_down: int,
        ext2_up: int,
        ext2_down: int,
        output_label: bool,
    ) -> None:
        self.ext1_up = ext1_up
        self.ext1_down = ext1_down
        self.ext2_up = ext2_up
        self.ext2_down = ext2_down
        self.seq_tokenizer = SeqTokenizer("PSACGT")
        self.two_mer_energy = TwoMerEnergy()
        self.energy_records = {}
        self.get_mh = None
        self.output_label = output_label
        self.ext_stem = "(((((((((.((((....))))...)))))))"

    def get_energy(self, examples: list[dict]) -> None:
        sp = subprocess.Popen(
            "RNAfold --noPS",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
        )
        fa_lines = []
        for example in examples:
            ref1 = example["ref1"]
            cut1 = example["cut1"]
            sgRNA = ref1[cut1 - 17 : cut1 + 3]
            sgRNA_scaffold = sgRNA + example["scaffold"]
            if sgRNA_scaffold not in self.energy_records:
                fa_lines.append(f">{len(fa_lines) // 2}")
                fa_lines.append(sgRNA)
                fa_lines.append(f">{len(fa_lines) // 2}")
                fa_lines.append(sgRNA_scaffold)
        stdout, _ = sp.communicate(input="\n".join(fa_lines).encode())
        for (
            _,
            sgRNA,
            sgRNA_energy,
            _,
            sgRNA_scaffold,
            sgRNA_scaffold_energy,
        ) in more_itertools.batched(stdout.decode().splitlines(), 6):
            dG = float(sgRNA_energy.split(" (")[1][:-2].strip())
            dG_binding_20 = self.two_mer_energy(sgRNA)
            dG_binding_7to20 = self.two_mer_energy(sgRNA[7:])
            align_seq = sgRNA_scaffold_energy.split(" (")[0]
            if align_seq[18 : 18 + len(self.ext_stem)] == self.ext_stem:
                stem = 1.0
            else:
                stem = 0.0
            self.energy_records[sgRNA_scaffold.replace("U", "T")] = [
                stem,
                dG,
                dG_binding_20,
                dG_binding_7to20,
            ]

    @torch.no_grad()
    def __call__(self, examples: list[dict]) -> dict:
        self.get_energy(examples)

        Xs, biological_inputs = [], []
        if self.output_label:
            observation_list = []

        for example in examples:
            ref = (
                example["ref1"][: example["cut1"]] + example["ref2"][example["cut2"] :]
            )
            cut = example["cut1"]
            self.assert_reference_length_and_cut(ref, cut)
            if (
                not self.get_mh
                or self.get_mh.ref1len != len(example["ref1"])
                or self.get_mh.ref2len != len(example["ref2"])
            ):
                self.get_mh = GetMH(
                    ref1len=len(example["ref1"]),
                    ref2len=len(example["ref2"]),
                )
            if self.output_label:
                mh_matrix, _, _, mh_rep_num = self.get_mh(
                    example["ref1"],
                    example["ref2"],
                    example["cut1"],
                    example["cut2"],
                    ext1=0,
                    ext2=0,
                )
                mh_idx = mh_matrix.nonzero()
                mh_val = mh_matrix[mh_idx]
                # construct observations
                observations = np.zeros(
                    (example["random_insert_uplimit"] + 2)
                    * (len(example["ref2"]) + 1)
                    * (len(example["ref1"]) + 1),
                    dtype=np.float32,
                )
                observations[example["ob_idx"]] = np.array(
                    example["ob_val"], dtype=np.float32
                )
                observations = observations.reshape(
                    example["random_insert_uplimit"] + 2,
                    len(example["ref2"]) + 1,
                    len(example["ref1"]) + 1,
                )
                # correct observations
                observations = self.get_mh.correct_observation(
                    observations, mh_matrix, mh_rep_num
                )
                # cumulate observations for all random insertion size
                observation = observations.sum(axis=0).flatten()
                # distribute count to all positions in single micro-homology diagonal
                observation[mh_idx] = observation[mh_idx] / (mh_val + 1)
                observation = observation.reshape(
                    len(example["ref2"]) + 1, len(example["ref1"]) + 1
                )
                # take the observation region based on model extension limits
                observation = observation[
                    example["cut2"]
                    - self.ext2_up : example["cut2"]
                    + self.ext2_down
                    + 1,
                    example["cut1"]
                    - self.ext1_up : example["cut1"]
                    + self.ext1_down
                    + 1,
                ]
                observation_list.append(torch.from_numpy(observation))

            sgRNA21mer = example["ref1"][example["cut1"] - 17 : example["cut1"] + 4]
            Xs.append(self.seq_tokenizer("S" + sgRNA21mer))

            gc_count = sgRNA21mer[:20].count("G") + sgRNA21mer[:20].count("C")
            gc_above_10 = float(gc_count > 10)
            gc_below_10 = float(gc_count < 10)
            Tm_global = Tm.Tm_NN(sgRNA21mer)
            Tm_5mer_end = Tm.Tm_NN(sgRNA21mer[15:21])
            Tm_8mer_middle = Tm.Tm_NN(sgRNA21mer[4:13])
            Tm_4mer_start = Tm.Tm_NN(sgRNA21mer[0:4])

            biological_inputs.append(
                self.energy_records[sgRNA21mer[:20] + example["scaffold"]]
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

        if self.output_label:
            return {
                "X": torch.from_numpy(np.stack(Xs)),
                "biological_input": torch.tensor(
                    biological_inputs, dtype=torch.float32
                ),
                "observation": torch.stack(observation_list),
            }
        return {
            "X": torch.from_numpy(np.stack(Xs)),
            "biological_input": torch.tensor(biological_inputs, dtype=torch.float32),
        }

    def assert_reference_length_and_cut(self, ref: str, cut: int) -> None:
        assert cut >= 17 and len(ref) - cut >= 4, f"ref is too short to contain 21mer"
        assert (
            cut >= self.ext1_up
            and cut >= self.ext2_up
            and len(ref) - cut >= self.ext1_down
            and len(ref) - cut >= self.ext2_down
        ), f"reference is too short to support extensions, ext1_up: {self.ext1_up}, ext1_down: {self.ext1_down}, ext2_up: {self.ext2_up}, ext2_down: {self.ext2_down}"

    def inference(self, examples: list[dict]) -> dict:
        assert not self.output_label, "inference cannot output count"
        for example in examples:
            ref, cut = example.pop("ref"), example.pop("cut")
            self.assert_reference_length_and_cut(ref, cut)
            assert "scaffold" in example, "scaffold not provided"
            example["ref1"] = example["ref2"] = ref
            example["cut1"] = example["cut2"] = cut
        return examples
