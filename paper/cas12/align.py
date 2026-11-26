import os
import pathlib
import sys
from contextlib import ExitStack

import pandas as pd
import py2bit
import sh
from Bio.Seq import Seq

from .download import load_meta


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

    df_site = pd.concat(
        [df_predict.assign(group="predict"), df_detect.assign(group="detect")],
        ignore_index=True,
    )

    # handle a mistake in the paper table
    df_site.loc[df_site["start"] == 27761526, ["start", "cut"]] = 27761525

    refs = []
    PAM_sgRNAs = []
    with py2bit.open(os.fspath(cfg["2bit"])) as tb:
        for chrom, start, end, cut, strand in zip(
            df_site["chrom"],
            df_site["start"],
            df_site["end"],
            df_site["cut"],
            df_site["strand"],
        ):
            ref = tb.sequence(chrom, cut - cfg["extend"], cut + cfg["extend"])

            PAM_sgRNA = tb.sequence(chrom, start, end)
            if strand == "-":
                ref = str(Seq(ref).reverse_complement())
                PAM_sgRNA = str(Seq(PAM_sgRNA).reverse_complement())
            refs.append(ref)
            PAM_sgRNAs.append(PAM_sgRNA)

    df_site["ref"] = refs
    df_site["PAM_sgRNA"] = PAM_sgRNAs

    df_site.to_csv(cfg["output_dir"] / "target_sites.csv", index=False)


def load_fastq(fastq_file: os.PathLike) -> pd.DataFrame:
    return pd.DataFrame(
        pd.read_csv(fastq_file, header=None, sep="\t").values.reshape(-1, 4),
        columns=["seqname", "query", "plus", "quality"],
    )


def split_fastq_to_target_sites(cfg: dict) -> None:
    df_site = pd.read_csv(cfg["output_dir"] / "target_sites.csv", header=0)
    for site in df_site["site"]:
        (cfg["output_dir"] / "sites" / site).mkdir(exist_ok=True, parents=True)

    df_meta = load_meta(cfg)
    for SRR, assay_site_cell in zip(df_meta["run"], df_meta["assay_site_cell"]):
        if "predicted" in assay_site_cell:
            df_site_slice = df_site.query("group == 'predict'")
        else:
            df_site_slice = df_site.query("group == 'detect'")

        for suffix in ["_1", "_2"]:
            print(SRR, suffix)
            df_fq = load_fastq(cfg["output_dir"] / "fastqs" / f"{SRR}{suffix}.fastq")
            if suffix == "_1":
                primers = df_site_slice["P1"]
            else:
                primers = df_site_slice["P2"]

            breakpoint()
            with ExitStack() as stack:
                fds = [
                    stack.enter_context(
                        open(
                            cfg["output_dir"] / "sites" / site / f"{SRR}{suffix}.fastq",
                            "w",
                        )
                    )
                    for site in df_site_slice["site"]
                ]
                for seqname, query, quality in zip(
                    df_fq["seqname"], df_fq["query"], df_fq["quality"]
                ):
                    for fd, primer in zip(fds, primers):
                        if query.startswith(primer):
                            fd.write(f"{seqname}\n{query}\n+\n{quality}\n")
                            break


def collect_rearr_input(cfg: dict) -> None:
    SRR2cas = (
        load_meta(cfg)
        .assign(
            cas=lambda df: df["assay_site_cell"].map(
                lambda assay_site_cell: "AsCpf1"
                if "AsCpf1" in assay_site_cell
                else "LbCpf1"
                if "LbCpf1" in assay_site_cell
                else "control",
            ),
        )[["run", "cas"]]
        .set_index(keys="run")["cas"]
    )

    df_site = pd.read_csv(cfg["output_dir"] / "target_sites.csv", header=0)
    with (
        open(cfg["output_dir"] / "rearr.ref", "w") as rd,
        open(cfg["output_dir"] / "AsCpf1.query", "w") as ad,
        open(cfg["output_dir"] / "LbCpf1.query", "w") as ld,
    ):
        ref_id = -1
        for site, ref in zip(df_site["site"], df_site["ref"]):
            # duplicate primers
            if site in cfg["black_list_sites"]:
                continue

            print(site)

            ref_id += 1
            ref1 = ref[: cfg["extend"] + cfg["short_extend"]]
            ref2 = ref[cfg["extend"] - cfg["short_extend"] :]
            rd.write(
                f"{0}\t{ref1}\t{cfg['extend']}\t{cfg['short_extend']}\t{ref2}\t{len(ref2)}\n"
            )
            rd.flush()
            df_AsCpf1 = []
            df_LbCpf1 = []
            for fastq_file in os.listdir(cfg["output_dir"] / "sites" / site):
                SRR = fastq_file.split("_")[0]
                if SRR2cas[SRR] not in cfg["cas_proteins"]:
                    # control
                    continue

                df_query = (
                    load_fastq(cfg["output_dir"] / "sites" / site / fastq_file)["query"]
                    .str.upper()
                    .value_counts()
                    .reset_index()
                )
                if "_2" in fastq_file:
                    df_query["query"] = df_query["query"].map(
                        lambda query: str(Seq(query).reverse_complement())
                    )

                if SRR2cas[SRR] == "AsCpf1":
                    df_AsCpf1.append(df_query)
                elif SRR2cas[SRR] == "LbCpf1":
                    df_LbCpf1.append(df_query)

            for df_queries, wd in zip([df_AsCpf1, df_LbCpf1], [ad, ld]):
                pd.concat(df_queries, ignore_index=True).groupby(
                    "query", as_index=False
                )["count"].sum().assign(ref_id=ref_id)[
                    ["query", "count", "ref_id"]
                ].to_csv(
                    wd,
                    sep="\t",
                    index=False,
                    header=False,
                )


def align_by_rearr(cfg: dict) -> None:
    ref_file = cfg["output_dir"] / "rearr.ref"
    direction_file = cfg["output_dir"] / "rearr.direct"
    with open(ref_file, "r") as rd, open(direction_file, "w") as dd:
        for line in rd:
            dd.write("up\n")
    for cas in cfg["cas_proteins"]:
        query_file = cfg["output_dir"] / f"{cas}.query"
        alg_file = cfg["output_dir"] / f"{cas}.alg"
        sh.bash(
            c=f"rearrangement < {query_file} 3< {ref_file} > {alg_file}",
            _out=sys.stdout,
            _err=sys.stderr,
        )

        correct_file = cfg["output_dir"] / f"{cas}.correct"
        with open(alg_file, "r") as fd:
            sh.gawk(
                "-f",
                "correct_micro_homology.awk",
                "--",
                ref_file,
                direction_file,
                _in=fd,
                _out=correct_file,
            )
