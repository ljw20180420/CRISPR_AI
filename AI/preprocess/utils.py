import numpy as np

# torch does not import opt_einsum as backend by default. import opt_einsum manually will enable it.
from torch.backends import opt_einsum
from einops import repeat, rearrange


class GetMH:
    def __init__(self, ref1len: int, ref2len: int) -> None:
        self.ref1len = ref1len
        self.ref2len = ref2len
        # diag_indices example for ref2len = 3 and ref1len = 2:
        # 6 9 11   row_indices 0 0 0   col_indices 0 1 2
        # 3 7 10               1 1 1               0 1 2
        # 1 4 8                2 2 2               0 1 2
        # 0 2 5                3 3 3               0 1 2
        # diag_indices = np.ravel_multi_index(
        #     multi_index=(
        #         tensor([3, 2, 3, 1, 2, 3, 0, 1, 2, 0, 1, 0]),
        #         tensor([0, 0, 1, 0, 1, 2, 0, 1, 2, 1, 2, 2])
        #     ),
        #     dims=(4, 3),
        # )
        row_indices = repeat(
            np.arange(self.ref2len + 1), "r2 -> r2 r1", r1=self.ref1len + 1
        )
        col_indices = repeat(
            np.arange(self.ref1len + 1), "r1 -> r2 r1", r2=self.ref2len + 1
        )
        self.diag_indices = np.ravel_multi_index(
            multi_index=(
                # row index
                np.concatenate(
                    [
                        row_indices.diagonal(offset)
                        for offset in range(-ref2len, ref1len + 1)
                    ]
                ),
                # col index
                np.concatenate(
                    [
                        col_indices.diagonal(offset)
                        for offset in range(-ref2len, ref1len + 1)
                    ]
                ),
            ),
            dims=(self.ref2len + 1, self.ref1len + 1),
        )

    def __call__(
        self, ref1: str, ref2: str, cut1: int, cut2: int, ext1: int, ext2: int
    ) -> tuple[np.ndarray]:
        assert (
            len(ref1) == self.ref1len and len(ref2) == self.ref2len
        ), "reference length does not fit"
        assert cut1 + ext1 <= len(ref1) and ext2 <= cut2, "extend too much"
        mh_matrix = np.pad(
            (
                rearrange(
                    np.frombuffer(ref1[: cut1 + ext1].encode(), dtype=np.int8),
                    "r1 -> 1 r1",
                )
                == rearrange(
                    np.frombuffer(ref2[cut2 - ext2 :].encode(), dtype=np.int8),
                    "r2 -> r2 1",
                )
            ).astype(int),
            pad_width=((cut2 - ext2, 1), (0, len(ref1) - cut1 - ext1 + 1)),
        )
        rep_num = np.diff(
            np.concatenate(
                (
                    np.array([-1], dtype=int),
                    np.where(np.diff(mh_matrix.flatten()[self.diag_indices]))[0],
                    np.array([(len(ref1) + 1) * (len(ref2) + 1) - 1], dtype=int),
                )
            )
        )
        rep_val = rep_num.copy()
        rep_val[0::2] = 0
        rep_num[1::2] = rep_num[1::2] + 1
        rep_num[2::2] = rep_num[2::2] - 1
        mh_matrix = mh_matrix.flatten()
        mh_matrix[self.diag_indices] = np.repeat(rep_val, rep_num)
        cum_rep_num = rep_num.cumsum()
        mh_idx_align_ref1 = self.diag_indices[cum_rep_num[1::2] - 1]
        mh_idx_align_ref2 = self.diag_indices[cum_rep_num[0:-1:2]]
        mh_rep_num = rep_num[1::2]
        return mh_matrix, mh_idx_align_ref1, mh_idx_align_ref2, mh_rep_num

    def correct_observation(
        self, observations: np.ndarray, mh_matrix: np.ndarray, mh_rep_num: np.ndarray
    ) -> tuple[np.ndarray]:
        mh_mask = (mh_matrix > 0)[self.diag_indices]
        for i, observation in enumerate(observations):
            observation = observation.flatten()
            counts = np.zeros(len(mh_rep_num), dtype=int)
            np.add.at(
                counts,
                np.repeat(np.arange(len(mh_rep_num)), mh_rep_num),
                observation[self.diag_indices][mh_mask],
            )
            observation[self.diag_indices[mh_mask]] = np.repeat(counts, mh_rep_num)
            observations[i] = observation.reshape(self.ref2len + 1, self.ref1len + 1)

        return observations
