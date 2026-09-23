"""Explicit public allowlist for the independent diffusion control."""
import hashlib,io,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
names={'LICENSE','NOTICE','pyproject.toml','scripts/diffusion_eval.py','scripts/prepare_cuda13.py','scripts/run_diffusion.sh','scripts/setup_diffusion_v2.sh','scripts/pod_deadline_v2.py','scripts/pod_start_bounded.sh','configs/diffusion-gpu-lock.txt','configs/diffusion-gpu-requirements.txt','research/snapshots/jevbench/public-eval-manifest.json','research/snapshots/jevbench/LICENSE','research/snapshots/jevbench/THIRD-PARTY.md'}
for directory,pattern in [('openjeff','*.py'),('research/vendor/djev','*'),('data/curriculum-v1','*.jsonl')]:
 names.update(str(p.relative_to(ROOT)) for p in (ROOT/directory).rglob(pattern) if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc',))
manifest=json.loads((ROOT/'research/snapshots/jevbench/public-eval-manifest.json').read_text())
names.update('research/snapshots/jevbench/'+e['path'] for e in manifest['files'])
# This control never trains; do not upload the 8,000-example training set.
names.discard('data/curriculum-v1/train.jsonl')
path=ROOT/'artifacts/openjeff-diffusion.tar.gz'
with tarfile.open(path,'w:gz') as archive:
 for n in sorted(names):
  raw=(ROOT/n).read_bytes();info=tarfile.TarInfo('openjeff-diffusion/'+n);info.size=len(raw);info.mode=0o644;info.mtime=0;archive.addfile(info,io.BytesIO(raw))
print(json.dumps({'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))
