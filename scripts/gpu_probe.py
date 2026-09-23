"""Execute a small real-model compatibility probe on a provisioned GPU.

This script does not provision, bill, or terminate cloud resources. Provider-side
termination must be configured independently. No private or card data is used.
Run from repository root: python -m scripts.gpu_probe --help
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

from openjeff.calibration import probabilities
from openjeff.evaluation import evaluate
from openjeff.probe_cases import generate_cases, to_request
from openjeff.prompting import compile_request

ROOT = Path(__file__).resolve().parents[1]


def model_entry(model_id):
    registry = json.loads((ROOT / "provenance/models.json").read_text())
    return next(m for m in registry["models"] if m["id"] == model_id)


def quantile(values, p):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)]


def tokenizer_check(cases):
    from transformers import AutoTokenizer, AutoConfig, AutoModelForImageTextToText
    results = []
    for model_id in ["google/gemma-4-E2B-it", "google/gemma-4-12B-it"]:
        path = ROOT / "artifacts/tokenizers" / model_id.replace("/", "--")
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        cfg = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        klass = AutoModelForImageTextToText._model_mapping[type(cfg)]
        lengths = []
        code_ids = set()
        for case in cases:
            compiled = compile_request(tok, to_request(case))
            lengths.append(len(compiled["input_ids"]))
            code_ids.update(compiled["code_ids"])
        results.append({"model": model_id, "revision": model_entry(model_id)["revision"],
                        "architecture": klass.__name__, "cases_compiled": len(cases),
                        "min_tokens": min(lengths), "max_tokens": max(lengths),
                        "candidate_token_ids": sorted(code_ids), "weights_loaded": False})
    return {"scope": "real_tokenizer_and_model_class_check_no_inference", "models": results}


def run_gpu(args, cases):
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from openjeff.adapters.hf import HFLabelScorer
    if not torch.cuda.is_available():
        raise RuntimeError("No functioning CUDA GPU; no download or inference attempted")
    entry = model_entry(args.model)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    snapshot = Path(snapshot_download(
        args.model, revision=entry["revision"], token=False, local_files_only=not args.download,
        allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json",
                        "generation_config.json", "tokenizer*", "special_tokens_map.json",
                        "chat_template.jinja", "LICENSE*", "NOTICE*", "README.md"]))
    manifest = []
    for path in sorted(snapshot.rglob("*")):
        if path.is_file():
            checksum = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    checksum.update(block)
            manifest.append({"path": str(path.relative_to(snapshot)), "bytes": path.stat().st_size,
                             "sha256": checksum.hexdigest()})
    (out / "model-files.json").write_text(json.dumps(
        {"model": args.model, "revision": entry["revision"], "files": manifest}, indent=2) + "\n")
    # Local-only model load enforces an explicit staging boundary.
    load_start = time.monotonic()
    scorer = HFLabelScorer(args.model, entry["revision"], max_tokens=8192)
    load_s = time.monotonic() - load_start
    for case in cases[:2]:
        scorer.score(to_request(case))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    rows, times, winners = [], [], {}
    score_path = out / "scores.jsonl"
    with score_path.open("w") as stream:
        for case in cases:
            if time.monotonic() - started > args.max_work_seconds:
                raise TimeoutError("Probe work deadline reached; provider expiry remains independent")
            request = to_request(case)
            torch.cuda.synchronize()
            t = time.perf_counter()
            scores = scorer.score(request)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t
            p = probabilities(scores)
            winner = max(range(len(p)), key=p.__getitem__)
            winners[case["id"]] = request.candidates[winner].id
            row = {k: case[k] for k in ("id", "group_id", "split", "family", "input_sha256", "target")}
            row.update(scorer_id=scorer.scorer_id, scores=scores, latency_s=elapsed)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            rows.append(row)
            times.append(elapsed)
    groups = {case["group_id"] for case in cases}
    flips = sum(winners[g + "-original"] != winners[g + "-reversed"] for g in groups)
    summary = {"scope": "compatibility_probe_not_JevBench_or_final_quality_evaluation",
               "model": args.model, "revision": entry["revision"], "scorer_id": scorer.scorer_id,
               "torch": torch.__version__, "transformers": transformers.__version__,
               "gpu": torch.cuda.get_device_name(), "load_seconds_excluding_download": load_s,
               "total_wall_seconds": time.monotonic() - started,
               "device_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "device_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
               "serial_latency_s": {"p50": quantile(times, .5), "p95": quantile(times, .95)},
               "option_order_disagreement": flips / len(groups), "metrics": evaluate(rows),
               "scores_sha256": hashlib.sha256(score_path.read_bytes()).hexdigest(),
               "provider_billing_termination_performed": False}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["google/gemma-4-E2B-it", "google/gemma-4-12B-it"], default="google/gemma-4-12B-it")
    parser.add_argument("--tokenizer-only", action="store_true")
    parser.add_argument("--download", action="store_true", help="Explicitly stage pinned public weights on the GPU host")
    parser.add_argument("--count-per-family", type=int, default=20)
    parser.add_argument("--max-work-seconds", type=int, default=2400)
    parser.add_argument("--out", default="reports/gpu-probe")
    args = parser.parse_args()
    if not 60 <= args.max_work_seconds <= 3600:
        parser.error("max-work-seconds must be in [60, 3600]")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    cases = generate_cases(args.count_per_family)
    if args.tokenizer_only:
        summary = tokenizer_check(cases)
        (ROOT / "reports/tokenizer-probe.json").write_text(json.dumps(summary, indent=2) + "\n")
    else:
        summary = run_gpu(args, cases)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
