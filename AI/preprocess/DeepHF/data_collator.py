import numpy as np
import torch
import more_itertools
import Bio.SeqUtils.MeltingTemp as Tm
import subprocess
from ..utils import MicroHomologyTool
from ...dataset.utils import SeqTokenizer


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
    preprocess = "DeepHF"

    def __init__(
        self,
        ext1_up: int,
        ext1_down: int,
        ext2_up: int,
        ext2_down: int,
    ) -> None:
        self.ext1_up = ext1_up
        self.ext1_down = ext1_down
        self.ext2_up = ext2_up
        self.ext2_down = ext2_down
        self.seq_tokenizer = SeqTokenizer("PSACGT")
        self.two_mer_energy = TwoMerEnergy()
        self.energy_records = {}
        self.ext_stem = "(((((((((.((((....))))...)))))))"
        self.micro_homology_tool = MicroHomologyTool()

    @torch.no_grad()
    def __call__(self, examples: list[dict], output_label: bool) -> dict:
        self._get_energy(examples)

        Xs, biological_inputs = [], []
        if output_label:
            cut1s, cut2s, observation_list = [], [], []

        for example in examples:
            ref = (
                example["ref1"][: example["cut1"]] + example["ref2"][example["cut2"] :]
            )
            cut = example["cut1"]
            self._assert_reference_length_and_cut(ref, cut)
            if output_label:
                cut1s.append(example["cut1"])
                cut2s.append(example["cut2"])
                mh_matrix, _, _, mh_rep_num = self.micro_homology_tool.get_mh(
                    example["ref1"],
                    example["ref2"],
                    example["cut1"],
                    example["cut2"],
                    ext1=0,
                    ext2=0,
                )
                observation = self.micro_homology_tool.get_observation(
                    example, mh_matrix, mh_rep_num, lefts=None, rights=None
                )
                observation_list.append(observation)

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

        if output_label:
            return {
                "input": {
                    "X": torch.from_numpy(np.stack(Xs)),
                    "biological_input": torch.tensor(
                        biological_inputs,
                        dtype=torch.float32,
                    ),
                },
                "label": {
                    "cut1": np.array(cut1s),
                    "cut2": np.array(cut2s),
                    "observation": torch.from_numpy(np.stack(observation_list)),
                },
            }
        return {
            "input": {
                "X": torch.from_numpy(np.stack(Xs)),
                "biological_input": torch.tensor(
                    biological_inputs,
                    dtype=torch.float32,
                ),
            },
        }

    def _get_energy(self, examples: list[dict]) -> None:
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

    def _assert_reference_length_and_cut(self, ref: str, cut: int) -> None:
        assert cut >= 17 and len(ref) - cut >= 4, f"ref is too short to contain 21mer"
        assert (
            cut >= self.ext1_up
            and cut >= self.ext2_up
            and len(ref) - cut >= self.ext1_down
            and len(ref) - cut >= self.ext2_down
        ), f"reference is too short to support extensions, ext1_up: {self.ext1_up}, ext1_down: {self.ext1_down}, ext2_up: {self.ext2_up}, ext2_down: {self.ext2_down}"
