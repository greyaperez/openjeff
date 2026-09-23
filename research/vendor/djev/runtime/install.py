"""Install a hash-checked vLLM composite in the pinned build image.

No model loading or cloud operations. All originals, downloaded sources and
candidate Python syntax are checked before the first package write.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import urllib.request

from .vision_patch import PINNED as VISION_INPUTS, patch_source, replace_once

HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "sources.json").read_text())
HELPER = "model_executor/models/djev_vision_serving.py"
ATTENTION = "v1/attention/ops/triton_unified_attention.py"
ATTENTION_OUTPUT_SHA = "be2162872224f671a9bea3e55d5dea73f0eeaa60db53f640022a49b7dcca267e"
MODIFIED_SOURCE_NOTICE = (
    b"# Modified by Djev contributors: source-pinned structured reads, exact-label "
    b"capacity, native image attention, and invariant dispatch integration.\n"
    b"# See the Djev repository NOTICE and runtime source manifest for attribution.\n"
)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def fetch_source(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=90) as response:
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError("Pinned source exceeds the expected size limit")
    return raw


def base_hashes() -> dict[str, str]:
    return {**MANIFEST["collaborators"], **{
        str(Path(name).relative_to("vllm")): hashes["base_sha256"]
        for name, hashes in MANIFEST["overlays"].items()}}


def build_sources(originals: dict[str, bytes], *, fetch=fetch_source) -> dict[str, bytes]:
    expected = base_hashes()
    if set(originals) != set(expected):
        raise ValueError("The base source set does not match the manifest")
    for name, digest in expected.items():
        if sha(originals[name]) != digest:
            raise ValueError("Pinned base source hash differs: " + name)
    staged = {}
    revision = MANIFEST["structured_read_commit"]
    for relative, hashes in MANIFEST["overlays"].items():
        raw = fetch(f"https://raw.githubusercontent.com/mmastrac/vllm/{revision}/{relative}")
        if sha(raw) != hashes["overlay_sha256"]:
            raise ValueError("Pinned structured-read source hash differs: " + relative)
        staged[str(Path(relative).relative_to("vllm"))] = raw

    # Request a complete closed set of exact label probabilities.
    text = staged["sampling_params.py"].decode()
    staged["sampling_params.py"] = replace_once(text, "MAX_LOGPROB_TOKEN_IDS = 128\n",
                                                "MAX_LOGPROB_TOKEN_IDS = 512\n").encode()
    model = "model_executor/models/diffusion_gemma.py"
    anchor = "@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)\ndef _compiled_sample_step(\n"
    budget = "import torch._dynamo.config as _djev_dynamo\n_djev_dynamo.recompile_limit = max(_djev_dynamo.recompile_limit, 64)\n\n"
    staged[model] = replace_once(staged[model].decode(), anchor, budget + anchor).encode()
    for name in VISION_INPUTS:
        staged[name] = patch_source(name, staged[name])
    staged[HELPER] = (HERE / "vision.py").read_bytes()
    staged[ATTENTION] = replace_once(originals[ATTENTION].decode(),
        "        and max_seqlen_q > 1\n",
        "        and (max_seqlen_q > 1 or is_batch_invariant)\n").encode()
    if sha(staged[ATTENTION]) != ATTENTION_OUTPUT_SHA:
        raise ValueError("Attention dispatch patch output differs")
    # Retain the upstream license/copyright headers and identify this composite
    # in each redistributed upstream file (Apache-2.0 section 4(b)).
    for name in staged:
        if name != HELPER:
            staged[name] = MODIFIED_SOURCE_NOTICE + staged[name]
    for name, raw in staged.items():
        compile(raw, name, "exec")
    return staged


def _inside(site: Path, name: str) -> Path:
    path = site / name
    if Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("Source path escapes the package")
    for parent in [path, *path.parents]:
        if parent == site:
            break
        if parent.is_symlink():
            raise ValueError("Source files cannot be symlinks")
    if not path.resolve().is_relative_to(site.resolve()):
        raise ValueError("Source path escapes the package")
    return path


def install(site: Path, *, fetch=fetch_source):
    site = site.resolve()
    versions = {name: importlib.metadata.version(name) for name in MANIFEST["packages"]}
    if versions != MANIFEST["packages"]:
        raise ValueError("Runtime package versions differ from the pinned manifest")
    originals = {name: _inside(site, name).read_bytes() for name in base_hashes()}
    if _inside(site, HELPER).exists():
        raise ValueError("Vision helper is already installed; use a clean base image")
    candidates = build_sources(originals, fetch=fetch)
    paths = {name: _inside(site, name) for name in candidates}
    provenance = _inside(site, "djev_sources.json")
    if provenance.exists():
        raise ValueError("Djev sources are already installed")
    proof = {"base_commit": MANIFEST["base_commit"],
             "structured_read_commit": MANIFEST["structured_read_commit"],
             "packages": versions, "files": {name: sha(raw) for name, raw in candidates.items()}}
    for name, raw in candidates.items():
        path = paths[name]
        path.write_bytes(raw)
        path.with_suffix(".pyc").unlink(missing_ok=True)
        for cached in (path.parent / "__pycache__").glob(path.stem + ".*.pyc"):
            cached.unlink()
    provenance.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    return proof


def main():
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise RuntimeError("vLLM is absent; use the documented pinned image")
    proof = install(Path(spec.origin).parent)
    print(json.dumps(proof, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
