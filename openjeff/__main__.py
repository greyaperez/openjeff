import argparse
import json
from pathlib import Path
from .calibration import Calibration, digest, fit_temperature
from .evaluation import evaluate
from .budget import budget_preflight


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="OpenJeff offline research scaffold; no model downloads")
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("calibrate")
    fit.add_argument("scores")
    fit.add_argument("--out", required=True)
    ev = commands.add_parser("evaluate")
    ev.add_argument("scores")
    ev.add_argument("--calibration", required=True)
    ev.add_argument("--out", required=True)
    budget = commands.add_parser("budget")
    for name, default in [("spent", "0"), ("committed", "0"), ("hourly-rate", "3.49"),
                          ("hours", "14"), ("overhead", "0"), ("cap", "200"), ("reserve", "50")]:
        budget.add_argument("--" + name, default=default)
    args = parser.parse_args()
    try:
        if args.command == "calibrate":
            artifact = fit_temperature(read_rows(args.scores))
            result = artifact.to_dict()
        elif args.command == "evaluate":
            artifact = Calibration(**json.loads(Path(args.calibration).read_text()))
            rows = read_rows(args.scores)
            artifact.validate_evaluation(rows)
            result = {"scope": "offline_score_replay_not_a_model_benchmark",
                      "calibration_id": artifact.artifact_id, "evaluation_data_sha256": digest(rows),
                      "uncalibrated": evaluate(rows),
                      "calibrated": evaluate(rows, artifact.temperature)}
        else:
            result = budget_preflight(**{k: v for k, v in vars(args).items() if k != "command"})
        encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.command == "budget":
            print(encoded, end="")
            if not result["within_planned_stop"]:
                parser.exit(2, "Planned spending stop exceeded. No cloud action performed.\n")
        else:
            Path(args.out).write_text(encoded)
            print(args.out)
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
