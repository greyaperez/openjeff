"""Package the validated diffusion study without overwriting the frozen AR release."""
import gzip,hashlib,io,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    validation=json.loads((ROOT/'reports/hybrid-v2-validation.json').read_text())
    if validation['status']!='passed':raise ValueError('Study validation must pass')
    names={'README.md','LICENSE','NOTICE','pyproject.toml','.gitignore','scripts/setup_diffusion_v2.sh','scripts/run_hybrid_v2.sh'}
    names.update(str(p.relative_to(ROOT)) for p in (ROOT/'scripts').glob('*.sh'))
    receipt=ROOT/'reports/hybrid-v2-cloud/h200-volume-backup.json'
    if receipt.exists():names.add(str(receipt.relative_to(ROOT)))
    patterns={'openjeff':'*.py','scripts':'*.py','tests':'*.py','examples':'*','configs':'*','docs':'*.md','model_cards':'*.md','provenance':'*.json','data/curriculum-v1':'*','research/snapshots':'*','research/vendor/djev':'*','runs/pilot-v1':'*','runs/hybrid-v2':'*','assets/branding':'*','reports/figures':'*','reports/hybrid-v2-cloud/completed':'*','reports/hybrid-v2-cloud/port-collision':'*'}
    for directory,pattern in patterns.items():
        for p in (ROOT/directory).rglob(pattern):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':names.add(str(p.relative_to(ROOT)))
    for name in ('spending.json','release-validation.json','RESULTS.md','HYBRID-RESULTS.md','diffusion-hybrid-v2.json','hybrid-v2-results.json','hybrid-v2-validation.json','hybrid-bundle-v2.json','runpod-image-config.json','shutdown-diagnostic-v2.json','diffusion-source.json','diffusion-attempt.json','public-validation.json','curriculum-overlap.json','hybrid-export-validation.json'):
        p=ROOT/'reports'/name
        if p.exists():names.add(str(p.relative_to(ROOT)))
    payloads={}
    for name in sorted(names):
        p=ROOT/name
        if p.is_symlink() or not p.resolve().is_relative_to(ROOT):raise ValueError('Unsafe archive path')
        payloads[name]=p.read_bytes()
    manifest=[{'path':n,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for n,b in payloads.items()]
    payloads['STUDY-MANIFEST.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    target=ROOT/'artifacts/OpenJeff-diffusion-study-v2.tar.gz'
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('wb') as f,gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as g,tarfile.open(fileobj=g,mode='w') as archive:
        for n,b in payloads.items():
            info=tarfile.TarInfo('OpenJeff-diffusion-study-v2/'+n);info.size=len(b);info.mode=0o644;info.mtime=0;archive.addfile(info,io.BytesIO(b))
    checksum=hashlib.sha256(target.read_bytes()).hexdigest();target.with_suffix(target.suffix+'.sha256').write_text(checksum+'  '+target.name+'\n')
    result={'path':str(target),'bytes':target.stat().st_size,'sha256':checksum,'files':len(payloads),'original_AR_release_preserved':True,'excluded':['payment credentials','private attachments','account credentials','foundation weights','virtual environments','raw billing receipts']}
    (ROOT/'reports/hybrid-study-package.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
