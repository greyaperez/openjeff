#!/bin/bash
set -euo pipefail
cd /workspace/openjeff-diffusion
finish() {
 code=$?
 trap - EXIT
 printf '%s\n' "$code" > hybrid-exit-status.txt
 if [ -f api-pid.txt ]; then kill "$(cat api-pid.txt)" 2>/dev/null || true; fi
 if [ -f model-pid.txt ]; then kill -- "-$(cat model-pid.txt)" 2>/dev/null || true; fi
 mkdir -p runs
 tar -czf /workspace/openjeff-hybrid-results.tar.gz runs ./*.log ./*status.txt ./*inventory* 2>/dev/null || true
 sha256sum /workspace/openjeff-hybrid-results.tar.gz > /workspace/openjeff-hybrid-results.sha256
 echo HYBRID_EXPORT_READY
}
trap finish EXIT
timeout 2400 bash scripts/setup_diffusion_v2.sh > setup-driver.log 2>&1
export CUDA_HOME="$PWD/cuda-toolkit"
export PATH="$PWD/.venv/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib64:${LIBRARY_PATH:-}"
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TOKENIZERS_PARALLELISM=false
.venv/bin/uv pip install --python .venv/bin/python --no-deps peft==0.21.0 accelerate==1.15.0 >> setup.log 2>&1
timeout 2700 python -u -m scripts.diffusion_hybrid_stage > diffusion-evaluation.log 2>&1
# Save the completed first stage before replacing its GPU process.
tar -czf /workspace/openjeff-diffusion-stage.tar.gz runs/hybrid-v2 ./*inventory* ./*status.txt ./*.log 2>/dev/null || true
sha256sum /workspace/openjeff-diffusion-stage.tar.gz > /workspace/openjeff-diffusion-stage.sha256
kill "$(cat api-pid.txt)" 2>/dev/null || true
kill -- "-$(cat model-pid.txt)" 2>/dev/null || true
# vLLM owns worker subprocesses; wait for its process group to release memory.
python - <<'PY'
import subprocess,time
for _ in range(60):
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if not text:break
    time.sleep(2)
else:raise RuntimeError('Diffusion GPU workers did not exit; refusing to load a second model')
PY
timeout 4500 python -u -m scripts.ar_hybrid_stage > ar-evaluation.log 2>&1
