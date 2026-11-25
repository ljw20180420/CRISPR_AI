import os
import pathlib

import pandas as pd
import py2bit
from Bio.Seq import Seq


def get_references(cfg: dict) -> None:
    primer_file = (
        cfg["output_dir"]
        / pathlib.Path(cfg["primer_file"].removeprefix("https://")).name
    )
    df = pd.read_excel(primer_file, header=None)
    df_predict = (
        df.loc[32:107]
        .reset_index(drop=True)
        .rename(
            columns={
                0: "site",
                1: "P1",
                2: "P2",
                3: "location",
            }
        )
        .assign(
            site=lambda df: df["site"]
            .str.removeprefix("To amplify ")
            .str.replace(" ", "_"),
            chrom=lambda df: "chr" + df["location"].str.split(":", expand=True)[0],
            genome_range=lambda df: df["location"].str.split(":", expand=True)[1],
            start=lambda df: df["genome_range"]
            .str.split("-", n=1, expand=True)[0]
            .astype(int),
            end=lambda df: df["genome_range"]
            .str.split("-", n=1, expand=True)[1]
            .str[:-1]
            .astype(int),
            strand=lambda df: df["location"].str[-1:],
            cut=lambda df: df["end"].where(df["strand"] == "+", df["start"]),
        )[["site", "P1", "P2", "chrom", "start", "end", "cut", "strand"]]
        .query("not site.str.contains('off-target')")
        .reset_index(drop=True)
    )

    df_detect = (
        df.loc[111:143]
        .reset_index(drop=True)
        .rename(
            columns={
                0: "site",
                1: "P1",
                2: "P2",
            }
        )
        .assign(
            location=lambda df: df["site"].str.rsplit("_", n=1, expand=True)[1],
            site=lambda df: df["site"].str.rsplit("_", n=1, expand=True)[0],
            chrom=lambda df: "chr" + df["location"].str.split(":", expand=True)[0],
            genome_range=lambda df: df["location"].str.split(":", expand=True)[1],
            start=lambda df: df["genome_range"]
            .str.split("-", n=1, expand=True)[0]
            .astype(int),
            end=lambda df: df["genome_range"]
            .str.split("-", n=1, expand=True)[1]
            .str[:-1]
            .astype(int),
            strand=lambda df: df["location"].str[-1:],
            cut=lambda df: df["end"].where(df["strand"] == "+", df["start"]),
        )[["site", "P1", "P2", "chrom", "start", "end", "cut", "strand"]]
        .assign(
            site=lambda df: df["site"]
            + "_d"
            + df.groupby("site").cumcount().astype(str)
        )
    )

    df_merge = pd.concat(
        [df_predict.assign(group="predict"), df_detect.assign(group="detect")],
        ignore_index=True,
    )

    # handle a mistake in the paper table
    df_merge.loc[df_merge["start"] == 27761526, ["start", "cut"]] = 27761525

    refs = []
    PAM_sgRNAs = []
    with py2bit.open(os.fspath(cfg["2bit"])) as tb:
        for chrom, start, end, cut, strand in zip(
            df_merge["chrom"],
            df_merge["start"],
            df_merge["end"],
            df_merge["cut"],
            df_merge["strand"],
        ):
            ref = tb.sequence(chrom, cut - cfg["extend"], cut + cfg["extend"])

            PAM_sgRNA = tb.sequence(chrom, start, end)
            if strand == "-":
                ref = str(Seq(ref).reverse_complement())
                PAM_sgRNA = str(Seq(PAM_sgRNA).reverse_complement())
            refs.append(ref)
            PAM_sgRNAs.append(PAM_sgRNA)

    df_merge["ref"] = refs
    df_merge["PAM_sgRNA"] = PAM_sgRNAs

    df_merge.to_csv(cfg["output_dir"] / "target_sites.csv", index=False)

def split_fastq_to_target_sites(cfg: dict) -> None
    pass
