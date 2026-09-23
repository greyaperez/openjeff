"""Expose NVIDIA's pinned pip CUDA 13.0 development files for this isolated pilot."""
from pathlib import Path
import json,subprocess,sysconfig
root=Path(sysconfig.get_paths()['purelib'])/'nvidia/cu13'
prefix=Path.cwd()/'cuda-toolkit'
prefix.mkdir(exist_ok=False)
for name,source in [('include',root/'include'),('bin',root/'bin'),('lib64',root/'lib')]:
    if not source.is_dir():raise RuntimeError(f'Missing CUDA package directory: {source}')
    (prefix/name).symlink_to(source,target_is_directory=True)
# The package may ship only versioned libraries; add development aliases locally.
# Use a separate link directory rather than modifying vendor package files.
(prefix/'lib64').unlink();(prefix/'lib64').mkdir()
for lib in (root/'lib').glob('*'):
    (prefix/'lib64'/lib.name).symlink_to(lib)
    if '.so.' in lib.name:
        alias=prefix/'lib64'/(lib.name.partition('.so.')[0]+'.so')
        if not alias.exists():alias.symlink_to(lib)
if not (prefix/'include/nvrtc.h').is_file():raise RuntimeError('NVRTC headers absent')
check=prefix/'nvrtc_check.c';check.write_text('#include <nvrtc.h>\n')
subprocess.run(['gcc','-fsyntax-only','-I'+str(prefix/'include'),str(check)],check=True,timeout=30)
version=subprocess.check_output([str(prefix/'bin/nvcc'),'--version'],text=True)
if 'release 13.0' not in version:raise RuntimeError('CUDA compiler version differs from pilot pin')
print(json.dumps({'cuda_home':str(prefix),'compiler':version,'headers_checked':True},indent=2))
