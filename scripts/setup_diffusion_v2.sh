#!/bin/bash
set -euo pipefail
cd /workspace/openjeff-diffusion
python -m venv /tmp/openjeff-venv
ln -s /tmp/openjeff-venv .venv
.venv/bin/python -m pip install uv > bootstrap.log 2>&1
timeout 1200 .venv/bin/uv pip install --python .venv/bin/python --no-cache -r configs/diffusion-gpu-lock.txt --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match > setup.log 2>&1
.venv/bin/uv pip install --python .venv/bin/python --no-deps ./research/vendor/djev >> setup.log 2>&1
.venv/bin/python scripts/prepare_cuda13.py > cuda-inventory.json
export CUDA_HOME="$PWD/cuda-toolkit"
export PATH="$PWD/.venv/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib64:${LIBRARY_PATH:-}"
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TOKENIZERS_PARALLELISM=false
export DJEV_MAX_MODEL_LEN=8192 DJEV_CANVAS=128 DJEV_COMPACT_CANVAS=1 DJEV_UPSTREAM=http://127.0.0.1:19001
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > gpu-inventory.csv
(cd research/vendor/djev && python -m runtime.install > ../../../install.log && python -m pytest -q > ../../../tests.log)
(cd research/vendor/djev && exec setsid python ../../../scripts/serve_diffusion_v2.py) > model.log 2>&1 &
printf '%s\n' "$!" > model-pid.txt
python - <<'PY'
import json,os,time,urllib.request
pid=int(open('model-pid.txt').read())
for attempt in range(180):
    os.kill(pid,0)
    try:
        with urllib.request.urlopen('http://127.0.0.1:19001/v1/models',timeout=3) as r:
            body=json.load(r)
            if any(m.get('id')=='dgemma' for m in body.get('data',[])):print('MODEL_READY');break
    except Exception:time.sleep(5)
else:raise TimeoutError('Model not ready within fifteen minutes')
PY
export DJEV_OFFLINE=1
python -m djev --host 127.0.0.1 --port 8000 > api.log 2>&1 &
printf '%s\n' "$!" > api-pid.txt
python - <<'PY'
import time,urllib.request
for attempt in range(30):
    try:
        with urllib.request.urlopen('http://127.0.0.1:8000/ready',timeout=10) as r:
            if r.status==200:print('API_READY');break
    except Exception:time.sleep(2)
else:raise TimeoutError('API readiness failed')
PY
printf '0\n' > setup-exit-status.txt
echo DIFFUSION_RUNTIME_READY
