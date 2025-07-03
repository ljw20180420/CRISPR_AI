import torch.nn.functional as F
import numpy as np
import torch


@torch.no_grad
def features_pairwise(features1, features2):
    return (features1.unsqueeze(-1) * features2.unsqueeze(-2)).flatten(start_dim=-2)


@torch.no_grad
def get_feature_LocalCutSiteSequence(ref, cut):
    return F.one_hot(
        torch.from_numpy(
            (np.frombuffer(ref[cut - 5 : cut + 4].encode(), dtype=np.int8) % 5)
            .clip(max=3)
            .astype(np.int64)
        ),
        num_classes=4,
    ).flatten()


@torch.no_grad
def get_feature_LocalCutSiteSeqMatches(ref, cut):
    offset1_bases = ref[cut - 2] + ref[cut - 1] * 2 + ref[cut] * 3 + ref[cut + 1] * 4
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


@torch.no_grad
def get_feature_LocalRelativeSequence(ref, cut, left, right, ins_seq):
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


@torch.no_grad
def get_feature_SeqMatches(ref, cut, left, right, ins_seq):
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


@torch.no_grad
def get_feature_I1or2Rpt(ref, cut, left, ins_seq):
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


@torch.no_grad
def getLeftMH(ref, cut, left, right, mh_max=16):
    left_mh = None
    for i in range(1, mh_max + 2):
        if i > mh_max or ref[cut + left - i] != ref[cut + right - i]:
            if left_mh is None:
                left_mh = i - 1
            else:
                left_mh_1 = i - 1
                break
    if left_mh == mh_max:
        left_mh_1 = mh_max
    return left_mh, left_mh_1


@torch.no_grad
def getRightMH(ref, cut, left, right, mh_max=16):
    right_mh = None
    for i in range(0, mh_max + 1):
        if i >= mh_max or ref[cut + left + i] != ref[cut + right + i]:
            if right_mh is None:
                right_mh = i
            else:
                right_mh_1 = i
                break
    if right_mh == mh_max:
        right_mh_1 = mh_max
    return right_mh, right_mh_1


@torch.no_grad
def get_feature_microhomology(ref, cut, left, right, ins_seq):
    if len(ins_seq) > 0:
        return [False] * 21
    left_mh, left_mh_1 = getLeftMH(ref, cut, left, right)
    right_mh, right_mh_1 = getRightMH(ref, cut, left, right)
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


def data_collator_single_example(
    example: dict, pre_calculated_features: tuple, output_count: bool
):
    (
        lefts,
        rights,
        inss,
        feature_DelSize,
        feature_InsSize,
        feature_DelLoc,
        feature_InsSeq,
        feature_fix,
    ) = pre_calculated_features
    if output_count:
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
        # the last 20 elements of lefts and rights correspond to insert_count
        count = torch.cat(
            [
                observation[
                    rights[:-20] + example["cut2"], lefts[:-20] + example["cut1"]
                ],
                torch.tensor(example["insert_count"][:20], dtype=torch.float64),
            ],
            dim=0,
        )

    cut = example["cut1"]
    ref = example["ref1"][: example["cut1"]] + example["ref2"][example["cut2"] :]
    (
        feature_I1or2Rpt,
        feature_LocalCutSiteSequence,
        feature_LocalCutSiteSeqMatches,
        feature_LocalRelativeSequence,
        feature_SeqMatches,
        feature_microhomology,
    ) = ([], [], [], [], [], [])
    for left, right, ins_seq in zip(lefts.tolist(), rights.tolist(), inss):
        feature_I1or2Rpt.append(get_feature_I1or2Rpt(ref, cut, left, ins_seq))
        feature_LocalCutSiteSequence.append(get_feature_LocalCutSiteSequence(ref, cut))
        feature_LocalCutSiteSeqMatches.append(
            get_feature_LocalCutSiteSeqMatches(ref, cut)
        )
        feature_LocalRelativeSequence.append(
            get_feature_LocalRelativeSequence(ref, cut, left, right, ins_seq)
        )
        feature_SeqMatches.append(
            get_feature_SeqMatches(ref, cut, left, right, ins_seq)
        )
        feature_microhomology.append(
            get_feature_microhomology(ref, cut, left, right, ins_seq)
        )
    feature_I1or2Rpt = torch.stack(feature_I1or2Rpt)
    feature_LocalCutSiteSequence = torch.stack(feature_LocalCutSiteSequence)
    feature_LocalCutSiteSeqMatches = torch.stack(feature_LocalCutSiteSeqMatches)
    feature_LocalRelativeSequence = torch.stack(feature_LocalRelativeSequence)
    feature_SeqMatches = torch.stack(feature_SeqMatches)
    feature_microhomology = torch.tensor(feature_microhomology)
    feature_var = torch.cat(
        [
            features_pairwise(
                feature_LocalCutSiteSequence,
                torch.cat([feature_InsSize, feature_DelSize], dim=-1),
            ),
            features_pairwise(
                torch.cat(
                    [feature_microhomology, feature_LocalRelativeSequence],
                    dim=-1,
                ),
                torch.cat(
                    [feature_DelSize, feature_DelLoc],
                    dim=-1,
                ),
            ),
            features_pairwise(
                torch.cat([feature_LocalCutSiteSeqMatches, feature_SeqMatches], dim=-1),
                feature_DelSize,
            ),
            features_pairwise(
                torch.cat(
                    [
                        feature_InsSeq,
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
    feature = torch.cat([feature_fix, feature_var], dim=-1)
    if output_count:
        return ref, cut, feature, count
    return ref, cut, feature


@torch.no_grad()
def data_collator(
    examples: list[dict],
    pre_calculated_features: tuple,
    output_count: bool,
) -> dict:
    refs, cuts, features = [], [], []
    if output_count:
        counts = []
    for example in examples:
        if output_count:
            ref, cut, feature, count = data_collator_single_example(
                example, pre_calculated_features, output_count
            )
        else:
            ref, cut, feature = data_collator_single_example(
                example, pre_calculated_features, output_count
            )
        refs.append(ref)
        cuts.append(cut)
        features.append(feature)
        if output_count:
            counts.append(count)
    if output_count:
        return {
            "ref": refs,
            "cut": cuts,
            "feature": torch.stack(features),
            "count": torch.stack(counts),
        }
    return {
        "ref": refs,
        "cut": cuts,
        "feature": torch.stack(features),
    }
