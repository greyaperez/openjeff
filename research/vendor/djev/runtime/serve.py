"""Start the BF16 model on loopback; run the Djev API separately."""
import json
import os

from djev.config import MODEL, MODEL_REVISION, get_max_model_len


def command():
    return ["vllm", "serve", MODEL, "--revision", MODEL_REVISION,
            "--dtype", "bfloat16", "--load-format", "safetensors",
            "--served-model-name", "dgemma", "--host", "127.0.0.1", "--port", "8001",
            "--diffusion-config", json.dumps({"canvas_length": 128}),
            "--max-logprobs", "512", "--enable-prefix-caching", "--async-scheduling",
            "--attention-backend", "TRITON_ATTN", "--max-num-seqs", "32",
            "--max-model-len", str(get_max_model_len()), "--gpu-memory-utilization", "0.85",
            "--kv-cache-dtype", "bfloat16", "--max-num-batched-tokens", "2048",
            "--limit-mm-per-prompt", json.dumps({"image": 6, "video": 0, "audio": 0}),
            "--override-generation-config", json.dumps({"max_new_tokens": None})]


def main():
    env = {**os.environ, "VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_BATCH_INVARIANT": "1",
           "OMP_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"}
    os.execvpe("vllm", command(), env)


if __name__ == "__main__":
    main()
