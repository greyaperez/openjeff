"""Frozen inputs and calibration/reporting helpers for the hybrid experiment."""
import hashlib,json
from pathlib import Path
from openjeff.contracts import Candidate,DecisionRequest
from scripts.jevbench_public import to_request
ROOT=Path(__file__).resolve().parents[1]
SPLITS=('development','calibration_fit','calibration_check','final','adversarial','production_sim','public')
def read(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]
def cases(split):
    if split!='public':
        for r in read(ROOT/f'data/curriculum-v1/{split}.jsonl'):
            raw=r['request'];req=DecisionRequest(raw['state'],raw['question'],tuple(Candidate(**c) for c in raw['candidates']))
            yield r,req,{'type':'choice','instructions':req.question,'criteria':{c.id:c.description for c in req.candidates}}
    else:
        for tier in ('easy','original','hard'):
            for task in read(ROOT/f'research/snapshots/jevbench/datasets/public/{tier}.jsonl'):
                req=to_request(task);expected=str(task['expected']) if task['question']['type']=='score' else task['expected']
                r={'id':task['id'],'group_id':'public/'+task['id'],'split':'public','family':task['family'],'tier':tier,
                    'target':task['labels'].index(expected),'input_sha256':hashlib.sha256(json.dumps({'state':task['state'],'question':task['question']},sort_keys=True).encode()).hexdigest()}
                yield r,req,task['question']
def core(row):return {k:row[k] for k in ('id','group_id','split','family','input_sha256','target')}
def finish_scores(out,method,metadata):
    from openjeff.calibration import fit_temperature
    from openjeff.evaluation import evaluate
    rows={s:read(out/f'{method}-{s}.jsonl') for s in SPLITS}
    cal=fit_temperature(rows['calibration_fit']);(out/f'{method}-calibration.json').write_text(json.dumps(cal.to_dict(),indent=2)+'\n')
    summary={'runtime':metadata,'temperature':cal.temperature,'splits':{}}
    for split,rs in rows.items():
        if split not in ('development','calibration_fit'):
            # Public benchmark is a test split in the calibration contract.
            cal.validate_evaluation([{**r,'split':'test'} for r in rs] if split=='public' else rs)
        summary['splits'][split]={'raw':evaluate(rs),'calibrated':evaluate(rs,cal.temperature)}
    (out/f'{method}-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary
