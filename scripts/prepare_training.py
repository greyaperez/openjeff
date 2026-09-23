"""Compile every curriculum request with the pinned real tokenizer, without weights."""
import json
from pathlib import Path
from transformers import AutoTokenizer
from openjeff.prompting import compile_request
from openjeff.probe_cases import to_request

def main():
    root = Path('data/curriculum-v1')
    tok = AutoTokenizer.from_pretrained('artifacts/tokenizers/google--gemma-4-12B-it', local_files_only=True)
    results = []
    for p in sorted(root.glob('*.jsonl')):
        lengths = []
        for line in p.read_text().splitlines():
            row = json.loads(line)
            c = compile_request(tok, to_request(row), max_tokens=1536)
            lengths.append(len(c['input_ids']))
        results.append({'split':p.stem, 'n':len(lengths), 'min_tokens':min(lengths), 'max_tokens':max(lengths)})
    Path('reports/training-tokenizer-check.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))

if __name__ == '__main__': main()
