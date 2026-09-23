"""Download public pinned sources and compile the composite without installing it."""
from concurrent.futures import ThreadPoolExecutor
import json

from .install import HELPER, MANIFEST, MODIFIED_SOURCE_NOTICE, base_hashes, build_sources, fetch_source, sha


def verify():
    base = MANIFEST["base_commit"]
    revision = MANIFEST["structured_read_commit"]
    urls = {name: f"https://raw.githubusercontent.com/vllm-project/vllm/{base}/vllm/{name}"
            for name in base_hashes()}
    overlay_urls = [f"https://raw.githubusercontent.com/mmastrac/vllm/{revision}/{name}"
                    for name in MANIFEST["overlays"]]
    all_urls = list(urls.values()) + overlay_urls
    with ThreadPoolExecutor(max_workers=6) as executor:
        downloaded = dict(zip(all_urls, executor.map(fetch_source, all_urls), strict=True))
    candidates = build_sources({name: downloaded[url] for name, url in urls.items()}, fetch=downloaded.__getitem__)
    if any(not raw.startswith(MODIFIED_SOURCE_NOTICE) for name, raw in candidates.items() if name != HELPER):
        raise ValueError("A redistributed upstream source lacks its modification notice")
    return {"verified_base_sources": len(urls), "compiled_candidate_sources": len(candidates),
            "files": {name: sha(raw) for name, raw in candidates.items()}}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))
