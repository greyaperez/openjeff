"""Download public source text and metadata for a reproducible research snapshot.

Never downloads weights, runs remote code, or sends local files.
"""
import hashlib
import json
import pathlib
import urllib.request
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1] / "research" / "snapshots"
ROOT.mkdir(parents=True, exist_ok=True)
records = []


def fetch(url, name):
    req = urllib.request.Request(url, headers={"User-Agent": "OpenJeff-research"})
    with urllib.request.urlopen(req, timeout=30) as response:
        data = response.read()
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    records.append({"url": url, "path": name, "sha256": hashlib.sha256(data).hexdigest()})
    return data


for repo, prefix, paths in [
    ("fstandhartinger/jevbench", "jevbench", ["README.md", "RESULTS-v1.2.md", "RESULTS-COMBINATIONS.md", "LICENSE", "THIRD-PARTY.md", "datasets/HARD-TIER.md", "docs/v1.2-additions-winnow.md", "docs/v1.2-additions-djev-thinking.md", "jevbench/composite_v13.py", "results/v1.2/jevbench-v1.2-results.json"]),
    ("Davipar/djev-dev", "djev", ["README.md", "LICENSE", "NOTICE", "docs/architecture.md", "docs/runtime.md", "djev/engine.py", "djev/labels.py", "runtime/sources.json"]),
]:
    info = json.loads(fetch(f"https://api.github.com/repos/{repo}", f"{prefix}/repository.json"))
    commit = json.loads(fetch(f"https://api.github.com/repos/{repo}/commits/{info['default_branch']}", f"{prefix}/commit.json"))
    sha = commit["sha"]
    tree = json.loads(fetch(f"https://api.github.com/repos/{repo}/git/trees/{sha}?recursive=1", f"{prefix}/tree.json"))
    print(prefix, sha)
    for path in paths:
        fetch(f"https://raw.githubusercontent.com/{repo}/{sha}/{path}", f"{prefix}/{path}")
    if prefix == "djev":
        print("DJEV FILES", *[x['path'] for x in tree['tree'] if x['path'].endswith(('.md','.py','.toml'))], sep="\n")

models = []
for model in ["google/gemma-4-E2B-it", "google/gemma-4-E4B-it", "google/gemma-4-12B-it", "google/gemma-4-26B-A4B-it", "google/gemma-4-31B-it", "google/diffusiongemma-26B-A4B-it", "allenai/Olmo-3-7B-Instruct", "allenai/Olmo-3.1-32B-Instruct", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"]:
    slug = model.replace("/", "--")
    info = json.loads(fetch(f"https://huggingface.co/api/models/{model}", f"models/{slug}/metadata.json"))
    sha = info["sha"]
    for path in ["README.md", "config.json"]:
        fetch(f"https://huggingface.co/{model}/resolve/{sha}/{path}", f"models/{slug}/{path}")
    models.append({"id": model, "revision": sha, "license_declared": info.get("cardData", {}).get("license"), "weights_downloaded": False, "weights_sha256": None})
    print(model, sha)

(ROOT.parent.parent / "provenance" / "models.json").write_text(json.dumps({"status": "metadata pins only; not a deployment approval", "models": models}, indent=2) + "\n")
(ROOT / "manifest.json").write_text(json.dumps({"retrieved_at": datetime.now(timezone.utc).isoformat(), "files": records}, indent=2) + "\n")
