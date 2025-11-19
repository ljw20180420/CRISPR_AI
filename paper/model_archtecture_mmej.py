#!/usr/bin/env python

import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import importlib.util
import sys


sys.path.insert(0, pathlib.Path(__file__).parent.parent.as_posix())
from AI.preprocess.utils import MicroHomologyTool

ref = "".join(np.random.choice(list("ACGT"), 50))
ref1 = ref[:31]
ref2 = ref[19:]
micro_homology_tool = MicroHomologyTool()
micro_homology_tool.reinitialize(ref1=ref1, ref2=ref2)
mh_matrix, _, _, _ = micro_homology_tool.get_mh(
    ref1=ref1, ref2=ref2, cut1=25, cut2=6, ext1=0, ext2=0
)
mh_matrix = mh_matrix.reshape(len(ref2) + 1, len(ref1) + 1)
min_mh = 2
mh_matrix[mh_matrix < min_mh] = 0
mh_facet = np.minimum(mh_matrix[:-1, :-1], mh_matrix[1:, 1:])

fig, ax = plt.subplots()
ax.imshow(
    mh_facet,
    cmap=LinearSegmentedColormap(
        name="white_red",
        segmentdata={
            "red": [(0.0, 1.0, 1.0), (1.0, 1.0, 1.0)],
            "green": [(0.0, 1.0, 1.0), (1.0, 0.0, 0.0)],
            "blue": [(0.0, 1.0, 1.0), (1.0, 0.0, 0.0)],
        },
        N=256,
    ),
)
ax.scatter(24.5, 5.5, c="black", marker="x", clip_on=False)
ax.scatter(
    np.random.choice(np.arange(-0.5, 31), 1),
    np.random.choice(np.arange(-0.5, 31), 1),
    c="black",
    marker="o",
    clip_on=False,
)


ax.set_xticks(
    ticks=np.arange(31),
    labels=list(ref1),
    fontdict={"family": "monospace", "color": "blue", "size": 10},
)
ax.set_yticks(
    ticks=np.arange(31),
    labels=list(ref2),
    fontdict={"family": "monospace", "color": "blue", "size": 10},
)

ax.spines["bottom"].set_color("green")
ax.spines["top"].set_color("green")
ax.spines["left"].set_color("green")
ax.spines["right"].set_color("green")
ax.tick_params(axis="both", left=False, bottom=False)
ax.set_xticks(np.arange(-0.5, 31, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 31, 1), minor=True)
ax.grid(which="minor", color="green", linestyle="-", linewidth=1)
ax.tick_params(which="minor", bottom=False, left=False)
fig.savefig("paper/figure/model_archtecture_mmej.pdf")
