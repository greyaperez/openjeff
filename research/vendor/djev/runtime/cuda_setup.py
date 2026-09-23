"""Expose development files already shipped in the pinned CUDA13 base image."""
from pathlib import Path
import subprocess
import tempfile


def ensure_cuda_development_files(cuda: Path, packaged_cuda: Path):
    include, libraries = cuda/'include', cuda/'lib64'
    source = packaged_cuda/'include'
    if not include.is_dir() or not libraries.is_dir() or not source.is_dir():
        raise RuntimeError('Pinned CUDA13 development directories are missing')
    linked_headers = linked_libraries = 0
    for header in sorted(source.iterdir()):
        destination = include/header.name
        if not destination.exists():
            destination.symlink_to(header)
            linked_headers += 1
    for library in sorted(libraries.glob('lib*.so.[0-9]*')):
        if not library.is_file():
            continue
        destination = libraries/(library.name.partition('.so.')[0]+'.so')
        if not destination.exists():
            destination.symlink_to(library)
            linked_libraries += 1
    if not (include/'nvrtc.h').is_file() or not (libraries/'libnvrtc.so').is_file():
        raise RuntimeError('CUDA13 NVRTC development files remain unavailable')
    return {'header_links': linked_headers, 'library_links': linked_libraries}


if __name__ == '__main__':
    import json
    cuda=Path('/usr/local/cuda')
    result=ensure_cuda_development_files(cuda,Path('/usr/local/lib/python3.12/dist-packages/nvidia/cu13'))
    with tempfile.TemporaryDirectory() as directory:
        probe=Path(directory)/'nvrtc_check.c'
        probe.write_text('#include <nvrtc.h>\n')
        subprocess.run(['gcc','-fsyntax-only','-I'+str(cuda/'include'),str(probe)],check=True,timeout=30)
    print(json.dumps(result))
