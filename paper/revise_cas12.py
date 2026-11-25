#!/usr/bin/env python

from cas12 import config, download

if __name__ == "__main__":
    cfg = config.cas12()

    # download.download_SRR(cfg)
    download.download_metadata(cfg)
