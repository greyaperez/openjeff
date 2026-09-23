import json
from pathlib import Path
from openjeff.curriculum import build_curriculum

if __name__ == "__main__":
    print(json.dumps(build_curriculum(Path(__file__).resolve().parents[1] / "data/curriculum-v1"), indent=2))
