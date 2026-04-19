FROM quay.io/condaforge/miniforge3:latest

WORKDIR /AI_env/genome
RUN wget https://genome-idx.s3.amazonaws.com/bt/hg19.1.bt2
RUN wget https://genome-idx.s3.amazonaws.com/bt/hg19.2.bt2
RUN wget https://genome-idx.s3.amazonaws.com/bt/hg19.3.bt2
RUN wget https://genome-idx.s3.amazonaws.com/bt/hg19.4.bt2
RUN wget https://genome-idx.s3.amazonaws.com/bt/hg19.rev.1.bt2
RUN wget https://genome-idx.s3.amazonaws.com/bt/hg19.rev.2.bt2
RUN wget https://hgdownload.cse.ucsc.edu/goldenpath/hg19/bigZips/hg19.2bit
WORKDIR /AI_env
COPY environment.yml ./
RUN conda env update -n base --file environment.yml && \
    conda clean -a && \
    rm environment.yml
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip cache purge
COPY ./ CRISPR_AI
ENV HF_HOME="/AI_env/hf_cache" \
    PYTHONPATH="/AI_env/CRISPR_AI/docker/common_ai" \
    GENOME="/AI_env/genome/hg19.2bit" \
    BOWTIE2_INDEX="/AI_env/genome/hg19" \
    device="cpu" \
    GRADIO_SERVER_NAME="0.0.0.0"
WORKDIR /AI_env/CRISPR_AI
CMD ["scripts/formal/app.sh"]
