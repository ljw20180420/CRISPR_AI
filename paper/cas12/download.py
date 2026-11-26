import sys

import pandas as pd
import sh


def download_metadata(cfg: dict):
    cfg["output_dir"].mkdir(exist_ok=True, parents=True)
    esearch = sh.esearch.bake(
        _long_sep=" ",
        _long_prefix="-",
    )
    efetch = sh.efetch.bake(
        _long_sep=" ",
        _long_prefix="-",
    )
    sh.xtract(
        "-pattern",
        "EXPERIMENT_PACKAGE",
        "-element",
        "RUN@accession",
        "EXPERIMENT@accession",
        "SAMPLE@accession",
        "-block",
        "SAMPLE_ATTRIBUTE",
        "-element",
        "TAG,VALUE",
        _in=efetch(
            format="xml",
            _in=esearch(
                db="sra",
                query=cfg["SRP"],
                _piped=True,
                _err=sys.stderr,
            ),
            _piped=True,
            _err=sys.stderr,
        ),
        _out=cfg["output_dir"] / f"{cfg['SRP']}_metadata.tsv",
        _err=sys.stderr,
    )

    sh.wget(
        cfg["primer_file"],
        directory_prefix=cfg["output_dir"],
        _err=sys.stderr,
    )


def load_meta(cfg: dict):
    df_meta = (
        pd.read_csv(
            cfg["output_dir"] / f"{cfg['SRP']}_metadata.tsv",
            sep="\t",
            names=[
                "run",
                "experiment",
                "sample",
                "isolate_key",
                "isolate",
                "age_key",
                "age",
                "biomaterial_provider_key",
                "biomaterial_provider",
                "sex_key",
                "sex",
                "tissue_key",
                "tissue",
                "assay_site_cell_key",
                "assay_site_cell",
                "BioSampleModel_key",
                "BioSampleModel",
            ],
        )
        .drop(
            columns=[
                "isolate_key",
                "age_key",
                "biomaterial_provider_key",
                "sex_key",
                "tissue_key",
                "assay_site_cell_key",
                "BioSampleModel_key",
            ]
        )
        .query("assay_site_cell.str.startswith('Targeted_')")
        .reset_index(drop=True)
    )

    return df_meta


def download_SRR(cfg: dict):
    (cfg["output_dir"] / "fastqs").mkdir(exist_ok=True, parents=True)
    df_meta = load_meta(cfg)
    for SRR in df_meta["run"]:
        sh.fasterq_dump(
            SRR,
            O=cfg["output_dir"] / "fastqs",
            t=cfg["output_dir"] / "fastqs",
            p=True,
            _out=sys.stdout,
            _err=sys.stderr,
        )
