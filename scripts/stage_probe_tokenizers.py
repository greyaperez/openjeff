"""Download ONLY public pinned tokenizer/config files. No weights or credentials."""
import hashlib
import json
import os
from pathlib import Path

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
from huggingface_hub import snapshot_download

root = Path(__file__).resolve().parents[1]
registry = json.loads((root / "provenance/models.json").read_text())
records = []
for entry in registry["models"]:
    if entry["id"] not in {"google/gemma-4-12B-it", "google/gemma-4-E2B-it"}:
        continue
    path = root / "artifacts/tokenizers" / entry["id"].replace("/", "--")
    snapshot_download(entry["id"], revision=entry["revision"], local_dir=path,
                      allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json",
                                      "special_tokens_map.json", "chat_template.jinja"], token=False)
    records.append({"model": entry["id"], "revision": entry["revision"],
                    "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in path.iterdir() if p.is_file()}})
(root / "reports/tokenizer-stage.json").write_text(json.dumps(records, indent=2) + "\n")
print("Staged two public tokenizers; no model weights downloaded.")

