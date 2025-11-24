FROM quay.io/condaforge/miniforge3:latest 

WORKDIR /AI_env
COPY requirements_conda.txt requirements.txt requirements_torch_pascal.txt ./
RUN conda install --file requirements_conda.txt && \
    conda clean -a
RUN pip install -r requirements.txt && \
    pip install -r requirements_torch_pascal.txt && \
    pip cache purge
RUN rm requirements_conda.txt requirements.txt requirements_torch_pascal.txt
COPY ./ CRISPR_AI
RUN wget https://github.com/ljw20180420/common_ai/archive/refs/tags/v1.0.8.tar.gz
RUN mkdir common_ai CRISPR_result && \
    tar zxf v1.0.8.tar.gz -C common_ai --strip-components 1 && \
    rm v1.0.8.tar.gz
ENV HF_HOME="/AI_env/hf_cache" \
    PYTHONPATH="/AI_env/common_ai" \
    OUTPUT_DIR="/AI_env/CRISPR_result" \
    device="cpu" \
    GRADIO_SERVER_NAME="0.0.0.0"
WORKDIR /AI_env/CRISPR_AI
CMD ["scripts/formal/app.sh"]
