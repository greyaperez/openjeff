"""Build an explicit allowlist of public project code for a GPU host."""
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    names = {"pyproject.toml", "LICENSE", "NOTICE", "provenance/models.json",
             "configs/probe-gpu-requirements.txt", "scripts/gpu_probe.py", "scripts/pod_watchdog.py",
             "docs/probe-ready.md"}
    for directory in ("openjeff", "tests"):
        names.update(str(p.relative_to(ROOT)) for p in (ROOT / directory).rglob("*.py"))
    payloads = {name: (ROOT / name).read_bytes() for name in sorted(names)}
    manifest = [{"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                for name, data in payloads.items()]
    payloads["bundle-manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    output = ROOT / "artifacts/openjeff-probe.tar.gz"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for name, data in payloads.items():
            item = tarfile.TarInfo("openjeff-probe/" + name)
            item.size, item.mode, item.mtime = len(data), 0o644, 0
            archive.addfile(item, io.BytesIO(data))
    report = {"scope": "public_code_only_upload_bundle", "path": str(output.relative_to(ROOT)),
              "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "bytes": output.stat().st_size, "files": manifest, "uploaded": False}
    (ROOT / "reports/probe-bundle.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
