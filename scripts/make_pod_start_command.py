"""Print a bounded supervisor command within Runpod's 4,000-character limit."""
import base64,hashlib,json,shlex,zlib
from pathlib import Path

def main():
    root=Path(__file__).resolve().parents[1]
    bundle=root/'artifacts/openjeff-hybrid-v2.tar.gz'
    source=(root/'scripts/pod_supervisor_v2.py').read_bytes()
    compressed=base64.b64encode(zlib.compress(source,9)).decode()
    command='python3 -u -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(compressed)+')))')+' '+hashlib.sha256(bundle.read_bytes()).hexdigest()
    if len(command)>4000:raise ValueError('Startup command exceeds provider limit')
    print(command)

if __name__=='__main__':main()
