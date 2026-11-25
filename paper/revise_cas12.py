#!/usr/bin/env python

from cas12 import align, config, download

if __name__ == "__main__":
    cfg = config.cas12()

    # download.download_metadata(cfg)
    # download.download_SRR(cfg)
    align.get_references(cfg)
