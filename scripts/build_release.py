"""Create a deterministic, public-data-only OpenJeff release archive."""
import gzip,hashlib,io,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    validation=json.loads((ROOT/'reports/release-validation.json').read_text())
    if validation['status']!='passed':raise ValueError('Release validation must pass')
    names={'artifacts/openjeff-training.tar.gz','artifacts/openjeff-eval-addon.tar.gz','README.md','LICENSE','NOTICE','pyproject.toml','.gitignore','scripts/run_diffusion.sh','scripts/pod_start_bounded.sh'}
    patterns={'openjeff':'*.py','scripts':'*.py','tests':'*.py','examples':'*','configs':'*','docs':'*.md','model_cards':'*.md',
        'provenance':'*.json','data/curriculum-v1':'*','research/snapshots':'*','research/vendor/djev':'*','runs/pilot-v1':'*','assets/branding':'*'}
    for directory,pattern in patterns.items():
        for p in (ROOT/directory).rglob(pattern):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':names.add(str(p.relative_to(ROOT)))
    for name in ('spending.json','release-validation.json','local-test-results.txt','training-tokenizer-check.json','training-bundle.json','diffusion-source.json','diffusion-source-check.txt','validation.md','RESULTS.md','published-public-comparison.json','diffusion-attempt.json','public-validation.json','curriculum-overlap.json'):
        p=ROOT/'reports'/name
        if p.exists():names.add(str(p.relative_to(ROOT)))
    for directory in ('reports/gpu-pilot-2026-09-23','reports/training-gpu-2026-09-23','reports/diffusion-gpu-2026-09-23','runs/diffusion-v1','reports/figures'):
        if (ROOT/directory).exists():names.update(str(p.relative_to(ROOT)) for p in (ROOT/directory).rglob('*') if p.is_file())
    payloads={}
    for name in sorted(names):
        p=ROOT/name
        if p.is_symlink() or not p.resolve().is_relative_to(ROOT):raise ValueError('Unsafe release path')
        payloads[name]=p.read_bytes()
    manifest=[{'path':n,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for n,b in payloads.items()]
    payloads['RELEASE-MANIFEST.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    target=ROOT/'artifacts/OpenJeff-0.1.0-pilot.tar.gz'
    with target.open('wb') as raw, gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as compressed, tarfile.open(fileobj=compressed,mode='w') as archive:
        for name,data in payloads.items():
            info=tarfile.TarInfo('OpenJeff-0.1.0-pilot/'+name);info.size=len(data);info.mode=0o644;info.mtime=0
            archive.addfile(info,io.BytesIO(data))
    report={'path':str(target),'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'files':len(payloads),
        'excluded':['payment information','account credentials','private attachments','virtual environments','base model weights']}
    (ROOT/'reports/release-package.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
