"""Public-only source/data/adapter bundle for automatic hybrid evaluation."""
import gzip,hashlib,io,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
names={'LICENSE','NOTICE','pyproject.toml','docs/diffusion-hybrid-protocol-v2.md','configs/diffusion-gpu-lock.txt','scripts/setup_diffusion_v2.sh','scripts/run_hybrid_v2.sh'}
for directory,pattern in [('openjeff','*.py'),('scripts','*.py'),('tests','*.py'),('research/vendor/djev','*'),('data/curriculum-v1','*.jsonl'),('research/snapshots/jevbench','*'),('runs/pilot-v1/adapter','*')]:
 for p in (ROOT/directory).rglob(pattern):
  if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':names.add(str(p.relative_to(ROOT)))
names.discard('data/curriculum-v1/train.jsonl')
payloads={n:(ROOT/n).read_bytes() for n in sorted(names)}
manifest=[{'path':n,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for n,b in payloads.items()]
payloads['HYBRID-BUNDLE-MANIFEST.json']=(json.dumps(manifest,indent=2)+'\n').encode()
path=ROOT/'artifacts/openjeff-hybrid-v2.tar.gz'
with path.open('wb') as f,gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as g,tarfile.open(fileobj=g,mode='w') as archive:
 for n,b in payloads.items():
  t=tarfile.TarInfo('openjeff-diffusion/'+n);t.size=len(b);t.mode=0o644;t.mtime=0;archive.addfile(t,io.BytesIO(b))
result={'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'files':len(payloads)}
(ROOT/'reports/hybrid-bundle-v2.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
