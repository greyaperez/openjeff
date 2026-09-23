"""Check exported public-benchmark scores against pinned source and probabilities."""
import hashlib,json,math,sys
from pathlib import Path
from types import SimpleNamespace
from openjeff.calibration import probabilities

ROOT=Path(__file__).resolve().parents[1]
BENCH=ROOT/'research/snapshots/jevbench'
def read(path):return [json.loads(x) for x in path.read_text().splitlines()]
def near(a,b):
    if not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10):raise AssertionError((a,b))
def main():
    sys.path.insert(0,str(BENCH))
    from jevbench.scoring import score_task
    from jevbench.metrics import ece_top_label,latency_summary
    manifest=json.loads((BENCH/'public-eval-manifest.json').read_text())
    for item in manifest['files']:
        assert hashlib.sha256((BENCH/item['path']).read_bytes()).hexdigest()==item['sha256']
    tasks={}
    for tier in ('easy','original','hard'):
        for task in read(BENCH/f'datasets/public/{tier}.jsonl'):
            assert task['id'] not in tasks
            task['tier']=tier;tasks[task['id']]=task
    run=ROOT/'runs/pilot-v1';summary=json.loads((run/'jevbench-public/summary.json').read_text())
    counts={}
    for model in ('base','adapter'):
        rows=read(run/f'jevbench-public/{model}.jsonl')
        assert len(rows)==len(tasks)==231 and {r['id'] for r in rows}==set(tasks)
        temperature=json.loads((run/f'{model}-calibration.json').read_text())['temperature']
        for row in rows:
            task=tasks[row['id']]
            assert row['expected']==task['expected'] and row['tier']==task['tier']
            assert row['family']==task['family']
            assert row['excluded']==bool(task.get('provenance',{}).get('exclude_reason'))
            probs=dict(zip(task['labels'],probabilities(row['scores'],temperature)))
            assert set(probs)==set(row['probs'])
            for label,value in probs.items():near(value,row['probs'][label])
            actual=score_task(probs,SimpleNamespace(**task))
            for key in ('valid','strict_valid','renormalized','correct','predicted'):assert actual[key]==row[key]
            if 'ordinal_ev' in actual:near(actual['ordinal_ev'],row['ordinal_ev'])
            gold=task.get('provenance',{}).get('gold_probs')
            if gold:near(sum(abs(probs[k]-gold[k]) for k in gold)/len(gold),row['gold_probability_mae'])
        for tier in ('easy','original','hard','all'):
            rs=[r for r in rows if tier=='all' or r['tier']==tier]
            scored=[r for r in rs if r['expected'] is not None and not r['excluded']]
            claimed=summary['models'][model][tier]
            assert len(rs)==claimed['attempts'] and len(scored)==claimed['scorable']
            near(sum(bool(r['correct']) for r in scored)/len(scored),claimed['accuracy'])
            near(ece_top_label([(max(r['probs'].values()),r['correct']) for r in scored if r['valid']])['ece'],claimed['ece_10_bins_valid_labeled'])
            assert sum(not r['valid'] for r in rs)==claimed['invalid']
            for key,value in latency_summary([r['latency_s'] for r in rs]).items():near(value,claimed['latency'][key])
        counts[model]={'rows':len(rows),'correct':sum(r['correct'] for r in rows)}
    result={'status':'passed','rows_verified':462,'benchmark_revision':manifest['revision'],'models':counts,
        'checks':['public source checksums','exact public ID coverage','source labels','frozen temperature probabilities','upstream task scoring','tier accuracy and calibration','latency quantiles']}
    (ROOT/'reports/public-validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
