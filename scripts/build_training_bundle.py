"""Allowlist public source and generated data; never includes secrets or base weights."""
import hashlib, io, json, tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    names={'pyproject.toml','LICENSE','NOTICE','provenance/models.json',
      'configs/probe-gpu-requirements.txt','configs/train-gpu-requirements.txt',
      'scripts/train_pilot.py','scripts/pod_watchdog.py','scripts/gpu_probe.py','scripts/build_curriculum.py'}
    for directory,pattern in [('openjeff','*.py'),('tests','*.py'),('data/curriculum-v1','*')]:
        names.update(str(p.relative_to(ROOT)) for p in (ROOT/directory).rglob(pattern) if p.is_file())
    payloads={n:(ROOT/n).read_bytes() for n in sorted(names)}
    manifest=[{'path':n,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for n,b in payloads.items()]
    payloads['bundle-manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    output=ROOT/'artifacts/openjeff-training.tar.gz'
    with tarfile.open(output,'w:gz') as archive:
        for n,b in payloads.items():
            t=tarfile.TarInfo('openjeff-training/'+n);t.size=len(b);t.mode=0o644;t.mtime=0
            archive.addfile(t,io.BytesIO(b))
    report={'path':str(output.relative_to(ROOT)),'bytes':output.stat().st_size,'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'files':manifest}
    (ROOT/'reports/training-bundle.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2))
if __name__=='__main__':main()
