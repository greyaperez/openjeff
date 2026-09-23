"""Evaluation-only add-on, uploaded after training has begun. No training changes."""
import hashlib,io,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
names={'scripts/jevbench_public.py','scripts/verify_pilot_service.py','research/snapshots/jevbench/public-eval-manifest.json','research/snapshots/jevbench/LICENSE','research/snapshots/jevbench/THIRD-PARTY.md'}
manifest=json.loads((ROOT/'research/snapshots/jevbench/public-eval-manifest.json').read_text())
names.update('research/snapshots/jevbench/'+e['path'] for e in manifest['files'])
path=ROOT/'artifacts/openjeff-eval-addon.tar.gz'
with tarfile.open(path,'w:gz') as archive:
 for n in sorted(names):
  raw=(ROOT/n).read_bytes();info=tarfile.TarInfo('openjeff-training/'+n);info.size=len(raw);info.mode=0o644;info.mtime=0;archive.addfile(info,io.BytesIO(raw))
print(json.dumps({'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))
