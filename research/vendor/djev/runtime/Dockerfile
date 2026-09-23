FROM vllm/vllm-openai:nightly-dee37d89115db4c94a820a79a78a7828e141c910

WORKDIR /opt/djev
COPY LICENSE NOTICE ./
COPY runtime/ runtime/
RUN python3 -m runtime.install && python3 -m runtime.cuda_setup
COPY pyproject.toml ./
COPY djev/ djev/
RUN python3 -m pip install --no-cache-dir '.[tokenizer]'

ENV VLLM_USE_V2_MODEL_RUNNER=1 VLLM_BATCH_INVARIANT=1 OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false HF_HOME=/cache/huggingface \
    VLLM_CACHE_ROOT=/cache/djev-public-bf16/vllm \
    TORCHINDUCTOR_CACHE_DIR=/cache/djev-public-bf16/torchinductor \
    TRITON_CACHE_DIR=/cache/djev-public-bf16/triton \
    CUDA_CACHE_PATH=/cache/djev-public-bf16/cuda

ENTRYPOINT ["python3", "-m", "runtime.serve"]
