FROM quay.io/condaforge/miniforge3:latest 

WORKDIR /AI_env
COPY environment.yml ./
RUN conda env update -n base --file environment.yml && \
    conda clean -a && \
    rm environment.yml
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip cache purge
COPY ./ CRISPR_AI
RUN wget https://github.com/ljw20180420/common_ai/archive/refs/tags/v1.0.9.tar.gz
RUN mkdir common_ai CRISPR_result && \
    tar zxf v1.0.9.tar.gz -C common_ai --strip-components 1 && \
    rm v1.0.9.tar.gz
ENV HF_HOME="/AI_env/hf_cache" \
    PYTHONPATH="/AI_env/common_ai" \
    OUTPUT_DIR="/AI_env/CRISPR_result" \
    GENOME="/AI_env/CRISPR_AI/docker/genome/hg19.2bit" \
    BOWTIE2_INDEX="/AI_env/docker/genome/hg19" \
    device="cpu" \
    GRADIO_SERVER_NAME="0.0.0.0"
WORKDIR /AI_env/CRISPR_AI
CMD ["scripts/formal/app.sh"]
