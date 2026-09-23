#!/bin/bash
set -euo pipefail
cd /workspace/openjeff-diffusion
MODEL_PID=''
API_PID=''
finish() {
    code=$?
    trap - EXIT
    if [ -n "$API_PID" ]; then kill "$API_PID" 2>/dev/null || true; fi
    if [ -n "$MODEL_PID" ]; then kill "$MODEL_PID" 2>/dev/null || true; fi
    printf '%s\n' "$code" > diffusion-exit-status.txt
    mkdir -p runs
    tar -czf /workspace/openjeff-diffusion-results.tar.gz runs ./*.log ./*status.txt ./*inventory* 2>/dev/null || true
    sha256sum /workspace/openjeff-diffusion-results.tar.gz > /workspace/openjeff-diffusion-results.sha256
    echo DIFFUSION_EXPORT_READY
}
trap finish EXIT
python -m venv .venv
.venv/bin/python -m pip install uv > bootstrap.log 2>&1
timeout 900 .venv/bin/uv pip install --python .venv/bin/python --no-cache -r configs/diffusion-gpu-lock.txt --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match > setup.log 2>&1
.venv/bin/uv pip install --python .venv/bin/python --no-deps ./research/vendor/djev >> setup.log 2>&1
.venv/bin/python scripts/prepare_cuda13.py > cuda-inventory.json
export CUDA_HOME="$PWD/cuda-toolkit"
export PATH="$PWD/.venv/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib64:${LIBRARY_PATH:-}"
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TOKENIZERS_PARALLELISM=false
export DJEV_MAX_MODEL_LEN=8192 DJEV_CANVAS=128 DJEV_COMPACT_CANVAS=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > gpu-inventory.csv
(cd research/vendor/djev && python -m runtime.install > ../../../install.log && python -m pytest -q > ../../../tests.log)
(cd research/vendor/djev && exec timeout 1800 python -m runtime.serve) > model.log 2>&1 &
MODEL_PID=$!
python - <<'PY'
import time,urllib.request
for attempt in range(120):
    try:
        with urllib.request.urlopen('http://127.0.0.1:8001/v1/models',timeout=3) as r:
            if r.status==200:print('MODEL_READY');break
    except Exception:time.sleep(5)
else:raise TimeoutError('Model did not become ready within ten minutes')
PY
export DJEV_OFFLINE=1
python -m djev --host 127.0.0.1 --port 8000 > api.log 2>&1 &
API_PID=$!
python - <<'PY'
import time,urllib.request
for attempt in range(30):
    try:
        with urllib.request.urlopen('http://127.0.0.1:8000/ready',timeout=10) as r:
            if r.status==200:print('API_READY');break
    except Exception:time.sleep(2)
else:raise TimeoutError('API readiness failed')
PY
timeout 1200 python -u -m scripts.diffusion_eval > evaluation.log 2>&1
