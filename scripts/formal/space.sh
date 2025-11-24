#!/bin/bash

# change to the dir of the script
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

printf "FROM ghcr.io/ljw20180420/crispr_ai:latest\n" > Dockerfile
hf repo create CRISPR_AI --repo-type space --space_sdk docker --exist-ok
hf upload CRISPR_AI Dockerfile --repo-type space
