"""Evaluate a separately running, pinned local Djev/DiffusionGemma backend."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import urllib.request
from types import SimpleNamespace
from openjeff.calibration import digest,fit_temperature,probabilities
from openjeff.evaluation import evaluate

ROOT=Path(__file__).resolve().parents[1]

def request(payload,timeout=120):
    req=urllib.request.Request('http://127.0.0.1:8000/v1/request',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as response:return json.load(response)

def extract(body,labels):
    answer=body['answers']['decision']
    p={'no':1-answer['noul'],'yes':answer['noul']} if answer['type']=='noul' else answer['probabilities']
    if set(p)!=set(labels) or any(not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in p.values()) or abs(sum(p.values())-1)>1e-6:
        raise ValueError('Invalid complete label distribution')
    return [math.log(max(p[k],1e-300)) for k in labels],p

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',default='runs/diffusion-v1');args=parser.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    runtime={'model':'google/diffusiongemma-26B-A4B-it','revision':'f7f5b7f5fa82ffc52addd066915886d497f5517b',
        'runtime_commit':'3ce907e6835212f27ee82b4cee9039198c4abe35','precision':'BF16','read_steps':1,'samples':1,
        'canvas_bound':128,'compact_canvas':True,'seed':0,'score_mode':'categorical','isolation':'joint',
        'log_probability_zero_floor':1e-300,'max_model_len':8192}
    import importlib.metadata
    import torch
    runtime['gpu']=torch.cuda.get_device_name()
    runtime['torch']=torch.__version__
    runtime['cuda']=torch.version.cuda
    runtime['vllm']=importlib.metadata.version('vllm')
    runtime['transformers']=importlib.metadata.version('transformers')
    import importlib.util
    source_manifest=Path(importlib.util.find_spec('vllm').origin).parent/'djev_sources.json'
    runtime['installed_source_manifest']=json.loads(source_manifest.read_text())
    identity=digest(runtime)
    options={'seed':0,'samples':1,'steps':1,'diagnostics':True,'score_mode':'categorical','isolation':'joint'}
    summary={'runtime':runtime,'scorer_id':identity,'scope':'local inference method; no new diffusion weights trained','splits':{}}
    started=time.monotonic()
    def deadline():
        if time.monotonic()-started>1800:raise TimeoutError('Diffusion evaluation work deadline')
    def score_split(split):
        rows=[]
        with (out/(split+'.jsonl')).open('w') as stream:
            for line in (ROOT/f'data/curriculum-v1/{split}.jsonl').read_text().splitlines():
                deadline();row=json.loads(line);r=row['request'];labels=[c['id'] for c in r['candidates']]
                payload={'state':r['state'],'questions':{'decision':{'type':'choice','instructions':r['question'],'criteria':{c['id']:c['description'] for c in r['candidates']}}},'options':options}
                t=time.perf_counter();body=request(payload);scores,p=extract(body,labels)
                item={k:row[k] for k in ('id','group_id','split','family','input_sha256','target')}
                item.update(scores=scores,probabilities_as_returned=p,scorer_id=identity,latency_s=time.perf_counter()-t,usage=body.get('usage'),diagnostics=body.get('diagnostics'))
                stream.write(json.dumps(item,allow_nan=False)+'\n');stream.flush();rows.append(item)
        return rows
    # Warmup is outside request latency metrics; calibration fit remains distinct.
    request({'state':{'value':1},'questions':{'decision':{'type':'choice','instructions':'Choose the value','criteria':{'one':'1','two':'2'}}},'options':options})
    development=score_split('development')
    summary['development_raw']=evaluate(development)
    fit=score_split('calibration_fit');cal=fit_temperature(fit)
    (out/'calibration.json').write_text(json.dumps(cal.to_dict(),indent=2)+'\n')
    summary['temperature']=cal.temperature
    for split in ('calibration_check','final','adversarial','production_sim'):
        scored=score_split(split);cal.validate_evaluation(scored)
        summary['splits'][split]={'raw':evaluate(scored),'calibrated':evaluate(scored,cal.temperature)}
        print(json.dumps({'event':'diffusion_split','split':split,'accuracy':summary['splits'][split]['raw']['accuracy']}),flush=True)
    sys.path.insert(0,str(ROOT/'research/snapshots/jevbench'))
    from jevbench.scoring import score_task
    from jevbench.metrics import ece_top_label,latency_summary
    bench=[]
    with (out/'jevbench-public.jsonl').open('w') as stream:
        for tier in ('easy','original','hard'):
            for line in (ROOT/f'research/snapshots/jevbench/datasets/public/{tier}.jsonl').read_text().splitlines():
                deadline();task=json.loads(line);q=dict(task['question']);labels=task['labels']
                if q['type']=='choice' and isinstance(q.get('criteria'),dict):q['criteria']={k:q['criteria'][k] for k in labels}
                t=time.perf_counter();body=request({'state':task['state'],'questions':{'decision':q},'options':options})
                scores,praw=extract(body,labels);p=dict(zip(labels,probabilities(scores,cal.temperature)))
                scored=score_task(p,SimpleNamespace(**task))
                item={'id':task['id'],'tier':tier,'family':task['family'],'expected':task['expected'],'scores':scores,'probabilities_as_returned':praw,'latency_s':time.perf_counter()-t,**scored}
                gold=task.get('provenance',{}).get('gold_probs')
                if gold:item['gold_probability_mae']=sum(abs(p[k]-gold[k]) for k in gold)/len(gold)
                stream.write(json.dumps(item,allow_nan=False)+'\n');stream.flush();bench.append(item)
    summary['public_benchmark']={}
    for tier in ('easy','original','hard','all'):
        rows=[r for r in bench if tier=='all' or r['tier']==tier];scored=[r for r in rows if r['expected'] is not None]
        gold=[r['gold_probability_mae'] for r in rows if 'gold_probability_mae' in r]
        summary['public_benchmark'][tier]={'attempts':len(rows),'scorable':len(scored),'accuracy':sum(bool(r['correct']) for r in scored)/len(scored),
            'ece_10_bins':ece_top_label([(max(r['probs'].values()),r['correct']) for r in scored])['ece'],
            'gold_probability_mae':sum(gold)/len(gold) if gold else None,'latency':latency_summary([r['latency_s'] for r in rows])}
    summary['wall_seconds']=time.monotonic()-started
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print('DIFFUSION_EVALUATION_COMPLETE',flush=True)

if __name__=='__main__':main()
