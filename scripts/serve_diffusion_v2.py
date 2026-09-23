"""Use the pinned runtime on a loopback port unused by the Runpod template."""
import os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research/vendor/djev'))
from runtime.serve import command

if __name__ == '__main__':
    args=command();args[args.index('--port')+1]='19001'
    env={**os.environ,'VLLM_USE_V2_MODEL_RUNNER':'1','VLLM_BATCH_INVARIANT':'1','OMP_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false'}
    os.execvpe('vllm',args,env)
