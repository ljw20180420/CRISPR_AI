#!/bin/bash

# refer to https://github.com/moby/moby/pull/47352 for proxy issue of docker
# refer to https://github.com/gradio-app/gradio/issues/4046 for proxy issue of gradio
# use -t to flush stdout, refer to https://stackoverflow.com/questions/29663459/why-doesnt-python-app-print-anything-when-run-in-a-detached-docker-container
# Both -t and -i for Ctrl-C to work as expected (stop instead of detach), refer to https://github.com/moby/moby/issues/2838#issuecomment-29205965
docker run --rm -it \
    -v $HF_HOME:/AI_env/hf_cache \
    -p 7860:7860 \
    ghcr.io/ljw20180420/crispr_ai:latest
