import torch.nn.functional as F
import numpy as np
import torch
from ..utils import MicroHomologyTool
from common_ai.generator import MyGenerator


class DataCollator:
    def __init__(self, max_del_size: int) -> None:
        self.max_del_size = max_del_size
        (
            self.lefts,
            self.rights,
            self.inss,
            self.feature_DelSize,
            self.feature_InsSize,
            self.feature_DelLoc,
            self.feature_InsSeq,
            self.feature_fix,
        ) = self._pre_calculation()
        self.micro_homology_tool = MicroHomologyTool()

    def _pre_calculation(self) -> tuple:
        lefts = np.concatenate(
            [np.arange(-DEL_SIZE, 1) for DEL_SIZE in range(self.max_del_size, -1, -1)]
            + [np.zeros(20, np.int64)]
        )
        rights = np.concatenate(
            [
                np.arange(0, DEL_SIZE + 1)
                for DEL_SIZE in range(self.max_del_size, -1, -1)
            ]
            + [np.zeros(20, np.int64)]
        )
        inss = (self.max_del_size + 2) * (self.max_del_size + 1) // 2 * [""] + [
            "A",
            "C",
            "G",
            "T",
            "AA",
            "AC",
            "AG",
            "AT",
            "CA",
            "CC",
            "CG",
            "CT",
            "GA",
            "GC",
            "GG",
            "GT",
            "TA",
            "TC",
            "TG",
            "TT",
        ]

        feature_DelSize = []
        for left, right, ins_seq in zip(lefts.tolist(), rights.tolist(), inss):
            dsize = right - left
            feature_DelSize.append(
                (len(ins_seq) == 0)
                & torch.tensor(
                    [
                        True,
                        dsize == 1,
                        dsize >= 2 and dsize < 4,
                        dsize >= 4 and dsize < 8,
                        dsize >= 8 and dsize < 13,
                        dsize >= 13,
                    ]
                )
            )
        feature_DelSize = torch.stack(feature_DelSize)

        feature_InsSize = torch.tensor(
            [
                [len(ins_seq) > 0, len(ins_seq) == 1, len(ins_seq) == 2]
                for ins_seq in inss
            ]
        )

        feature_DelLoc = []
        for left, right, ins_seq in zip(lefts.tolist(), rights.tolist(), inss):
            if len(ins_seq) > 0:
                feature_DelLoc.append([False] * 18)
                continue
            feature_DelLoc.append(
                [
                    left == 0,
                    left == -1,
                    left == -2,
                    left > -2 and left <= -5,
                    left > -5 and left <= -9,
                    left > -9 and left <= -14,
                    left > -14 and left <= -29,
                    left < -29,
                    left >= 1,
                    right == 0,
                    right == 1,
                    right == 2,
                    right > 2 and right <= 5,
                    right > 5 and right <= 9,
                    right > 9 and right <= 14,
                    right > 14 and right <= 29,
                    right < 0,
                    right > 30,
                ]
            )
        feature_DelLoc = torch.tensor(feature_DelLoc)

        feature_InsSeq = torch.cat(
            [
                torch.full(
                    (
                        (self.max_del_size + 2) * (self.max_del_size + 1) // 2,
                        20,
                    ),
                    False,
                ),
                torch.eye(20, dtype=torch.bool),
            ]
        )

        feature_InsLoc = []
        for left, ins_seq in zip(lefts.tolist(), inss):
            if len(ins_seq) == 0:
                feature_InsLoc.append([False] * 5)
                continue
            feature_InsLoc.append(
                [left == 0, left == -1, left == -2, left < -2, left >= 1]
            )
        feature_InsLoc = torch.tensor(feature_InsLoc)

        feature_fix = torch.cat(
            [
                self._features_pairwise(feature_DelSize, feature_DelLoc),
                feature_InsSize,
                feature_DelSize,
                feature_DelLoc,
                feature_InsLoc,
                feature_InsSeq,
            ],
            dim=-1,
        )

        return (
            lefts,
            rights,
            inss,
            feature_DelSize,
            feature_InsSize,
            feature_DelLoc,
            feature_InsSeq,
            feature_fix,
        )

    def _features_pairwise(
        self, features1: torch.Tensor, features2: torch.Tensor
    ) -> torch.Tensor:
        return (features1.unsqueeze(-1) * features2.unsqueeze(-2)).flatten(start_dim=-2)

    @torch.no_grad()
    def __call__(
        self, examples: list[dict], output_label: bool, my_generator: MyGenerator
    ) -> dict:
        features = []
        if output_label:
            counts, observation_list = [], []
        for example in examples:
            cut = example["cut1"]
            ref = (
                example["ref1"][: example["cut1"]] + example["ref2"][example["cut2"] :]
            )
            self._assert_reference_length_and_cut(ref, cut)
            if output_label:
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
                # the last 20 elements of lefts and rights correspond to insert_count
                count = torch.cat(
                    [
                        torch.from_numpy(
                            observation[
                                self.rights[:-20] + example["cut2"],
                                self.lefts[:-20] + example["cut1"],
                            ],
                        ),
                        torch.tensor(example["insert_count"][:20], dtype=torch.float32),
                    ],
                    dim=0,
                )
                counts.append(count)

            (
                feature_I1or2Rpt,
                feature_LocalCutSiteSequence,
                feature_LocalCutSiteSeqMatches,
                feature_LocalRelativeSequence,
                feature_SeqMatches,
                feature_microhomology,
            ) = ([], [], [], [], [], [])
            for left, right, ins_seq in zip(
                self.lefts.tolist(), self.rights.tolist(), self.inss
            ):
                feature_I1or2Rpt.append(
                    self._get_feature_I1or2Rpt(ref, cut, left, ins_seq)
                )
                feature_LocalCutSiteSequence.append(
                    self._get_feature_LocalCutSiteSequence(ref, cut)
                )
                feature_LocalCutSiteSeqMatches.append(
                    self._get_feature_LocalCutSiteSeqMatches(ref, cut)
                )
                feature_LocalRelativeSequence.append(
                    self._get_feature_LocalRelativeSequence(
                        ref, cut, left, right, ins_seq
                    )
                )
                feature_SeqMatches.append(
                    self._get_feature_SeqMatches(ref, cut, left, right, ins_seq)
                )
                feature_microhomology.append(
                    self._get_feature_microhomology(ref, cut, left, right, ins_seq)
                )
            feature_I1or2Rpt = torch.stack(feature_I1or2Rpt)
            feature_LocalCutSiteSequence = torch.stack(feature_LocalCutSiteSequence)
            feature_LocalCutSiteSeqMatches = torch.stack(feature_LocalCutSiteSeqMatches)
            feature_LocalRelativeSequence = torch.stack(feature_LocalRelativeSequence)
            feature_SeqMatches = torch.stack(feature_SeqMatches)
            feature_microhomology = torch.tensor(feature_microhomology)
            feature_var = torch.cat(
                [
                    self._features_pairwise(
                        feature_LocalCutSiteSequence,
                        torch.cat([self.feature_InsSize, self.feature_DelSize], dim=-1),
                    ),
                    self._features_pairwise(
                        torch.cat(
                            [feature_microhomology, feature_LocalRelativeSequence],
                            dim=-1,
                        ),
                        torch.cat(
                            [self.feature_DelSize, self.feature_DelLoc],
                            dim=-1,
                        ),
                    ),
                    self._features_pairwise(
                        torch.cat(
                            [feature_LocalCutSiteSeqMatches, feature_SeqMatches], dim=-1
                        ),
                        self.feature_DelSize,
                    ),
                    self._features_pairwise(
                        torch.cat(
                            [
                                self.feature_InsSeq,
                                feature_LocalCutSiteSequence,
                                feature_LocalCutSiteSeqMatches,
                            ],
                            dim=-1,
                        ),
                        feature_I1or2Rpt,
                    ),
                    feature_I1or2Rpt,
                    feature_LocalCutSiteSequence,
                    feature_LocalCutSiteSeqMatches,
                    feature_LocalRelativeSequence,
                    feature_SeqMatches,
                    feature_microhomology,
                ],
                dim=-1,
            ).to(torch.float32)
            feature = torch.cat([self.feature_fix, feature_var], dim=1)
            features.append(feature)

        if output_label:
            return {
                "input": {
                    "feature": torch.stack(features),
                },
                "label": {
                    "count": torch.stack(counts),
                    "observation": torch.from_numpy(np.stack(observation_list)),
                },
            }
        return {
            "input": {
                "feature": torch.stack(features),
            },
        }

    def _get_feature_LocalCutSiteSequence(self, ref: str, cut: int) -> torch.Tensor:
        return F.one_hot(
            torch.from_numpy(
                (np.frombuffer(ref[cut - 5 : cut + 4].encode(), dtype=np.int8) % 5)
                .clip(max=3)
                .astype(np.int64)
            ),
            num_classes=4,
        ).flatten()

    def _get_feature_LocalCutSiteSeqMatches(self, ref: str, cut: int) -> torch.Tensor:
        offset1_bases = (
            ref[cut - 2] + ref[cut - 1] * 2 + ref[cut] * 3 + ref[cut + 1] * 4
        )
        offset2_bases = (
            ref[cut - 3]
            + ref[cut - 3 : cut - 1]
            + ref[cut - 3 : cut]
            + ref[cut - 3 : cut + 1]
        )
        return (
            F.one_hot(
                torch.from_numpy(
                    (np.frombuffer(offset1_bases.encode(), dtype=np.int8) % 5)
                    .clip(max=3)
                    .astype(np.int64)
                ),
                num_classes=4,
            ).flatten()
            * F.one_hot(
                torch.from_numpy(
                    (np.frombuffer(offset2_bases.encode(), dtype=np.int8) % 5)
                    .clip(max=3)
                    .astype(np.int64)
                ),
                num_classes=4,
            ).flatten()
        )

    def _get_feature_LocalRelativeSequence(
        self, ref: str, cut: int, left: int, right: int, ins_seq: str
    ) -> torch.Tensor:
        if len(ins_seq) > 0:
            return torch.zeros(48, dtype=torch.int64)
        return torch.cat(
            [
                F.one_hot(
                    torch.from_numpy(
                        (
                            np.frombuffer(
                                ref[cut + left - 3 : cut + left + 3].encode(),
                                dtype=np.int8,
                            )
                            % 5
                        )
                        .clip(max=3)
                        .astype(np.int64)
                    ),
                    num_classes=4,
                ).flatten(),
                F.one_hot(
                    torch.from_numpy(
                        (
                            np.frombuffer(
                                ref[cut + right - 3 : cut + right + 3].encode(),
                                dtype=np.int8,
                            )
                            % 5
                        )
                        .clip(max=3)
                        .astype(np.int64)
                    ),
                    num_classes=4,
                ).flatten(),
            ]
        )

    def _get_feature_SeqMatches(
        self, ref: str, cut: int, left: int, right: int, ins_seq: str
    ) -> torch.Tensor:
        if len(ins_seq) > 0:
            return torch.zeros(72, dtype=torch.int64)
        return F.one_hot(
            torch.from_numpy(
                (
                    np.frombuffer(
                        ref[cut + left - 3 : cut + left + 3].encode(), dtype=np.int8
                    )[:, None]
                    == np.frombuffer(
                        ref[cut + right - 3 : cut + right + 3].encode(), dtype=np.int8
                    )
                ).astype(np.int64)
            ),
            num_classes=2,
        ).flatten()

    def _get_feature_I1or2Rpt(
        self, ref: str, cut: int, left: int, ins_seq: str
    ) -> torch.Tensor:
        if len(ins_seq) == 0:
            return torch.full((4,), False)
        return torch.tensor(
            [
                ins_seq == ref[cut - 1],
                len(ins_seq) == 1 and ins_seq != ref[cut - 1],
                ins_seq == (ref[cut - 1] * 2),
                len(ins_seq) == 2 and ins_seq != (ref[cut - 1] * 2),
            ]
        ).logical_and(torch.tensor(left == 0))

    def _get_feature_microhomology(
        self, ref: str, cut: int, left: int, right: int, ins_seq: str
    ) -> list[bool]:
        if len(ins_seq) > 0:
            return [False] * 21
        left_mh, left_mh_1 = self._getLeftMH(ref, cut, left, right)
        right_mh, right_mh_1 = self._getRightMH(ref, cut, left, right)
        return [
            left_mh == 1,
            right_mh == 1,
            left_mh == 2,
            right_mh == 2,
            left_mh == 3,
            right_mh == 3,
            left_mh_1 == 3,
            right_mh_1 == 3,
            left_mh >= 4 and left_mh < 7,
            right_mh >= 4 and right_mh < 7,
            left_mh_1 >= 4 and left_mh_1 < 7,
            right_mh_1 >= 4 and right_mh_1 < 7,
            left_mh >= 7 and left_mh < 11,
            right_mh >= 7 and right_mh < 11,
            left_mh_1 >= 7 and left_mh_1 < 11,
            right_mh_1 >= 7 and right_mh_1 < 11,
            left_mh >= 11 and left_mh < 16,
            right_mh >= 11 and right_mh < 16,
            left_mh_1 >= 11 and left_mh_1 < 16,
            right_mh_1 >= 11 and right_mh_1 < 16,
            left_mh == 0
            or left_mh >= 16
            and right_mh == 0
            or right_mh >= 16
            and left_mh_1 == 0
            or left_mh_1 >= 16
            and right_mh_1 == 0
            or right_mh_1 >= 16,
        ]

    def _getLeftMH(
        self, ref: str, cut: int, left: int, right: int, mh_max: int = 16
    ) -> tuple[int]:
        left_mh = None
        for i in range(1, mh_max + 2):
            if (
                i > mh_max
                or cut + left - i < 0
                or ref[cut + left - i] != ref[cut + right - i]
            ):
                if left_mh is None:
                    left_mh = i - 1
                else:
                    left_mh_1 = i - 1
                    break
        if left_mh == mh_max or cut + left - left_mh == 0:
            left_mh_1 = left_mh
        return left_mh, left_mh_1

    def _getRightMH(
        self, ref: str, cut: int, left: int, right: int, mh_max=16
    ) -> tuple[int]:
        right_mh = None
        for i in range(0, mh_max + 1):
            if (
                i >= mh_max
                or cut + right + i >= len(ref)
                or ref[cut + left + i] != ref[cut + right + i]
            ):
                if right_mh is None:
                    right_mh = i
                else:
                    right_mh_1 = i
                    break
        if right_mh == mh_max or cut + right + right_mh == len(ref):
            right_mh_1 = right_mh
        return right_mh, right_mh_1

    def _assert_reference_length_and_cut(self, ref: str, cut: int) -> None:
        assert (
            cut >= self.max_del_size
        ), f"ref upstream to cut ({cut}) is less than max_del_size ({self.max_del_size}), extend ref to upstream"
        assert (
            len(ref) - cut >= self.max_del_size
        ), f"ref downstream to cut ({len(ref) - cut}) is less than max_del_size ({self.max_del_size}), extend ref to downstream"
