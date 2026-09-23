"""Frozen public-subset diagnostic, never an official JevBench composite or rank."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from openjeff.calibration import probabilities
from openjeff.contracts import Candidate,DecisionRequest
from openjeff.prompting import compile_request
from openjeff.training import collate,candidate_logits,adapter_digest

ROOT=Path(__file__).resolve().parents[1]
BENCH=ROOT/'research/snapshots/jevbench'

def to_request(task):
    question=task['question'];criteria=question.get('criteria')
    options=[]
    for label in task['labels']:
        if isinstance(criteria,list):description=criteria[int(label)]
        elif isinstance(criteria,dict):description=criteria.get({'no':'false','yes':'true'}.get(label,label) if question['type']=='noul' else label,label)
        else:description=label
        options.append(Candidate(label,f'{label}: {description}'))
    # The answer, rationale, author metadata, and gold probabilities are never in the prompt.
    return DecisionRequest(task['state'],question['instructions'],tuple(options))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',default='runs/pilot-v1');args=parser.parse_args()
    run=Path(args.run)
    release=json.loads((run/'release.json').read_text())
    freeze=json.loads((run/'freeze.json').read_text())
    if release['package_adapter_sha256']!=adapter_digest(run/'adapter'):raise ValueError('Frozen adapter mismatch')
    import torch
    from transformers import AutoTokenizer,AutoModelForImageTextToText
    from peft import PeftModel
    sys.path.insert(0,str(BENCH))
    from jevbench.scoring import score_task
    from jevbench.metrics import ece_top_label,latency_summary
    manifest=json.loads((BENCH/'public-eval-manifest.json').read_text())
    for entry in manifest['files']:
        if hashlib.sha256((BENCH/entry['path']).read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Benchmark checksum mismatch')
    tokenizer=AutoTokenizer.from_pretrained(release['model'],revision=release['revision'],local_files_only=True)
    base=AutoModelForImageTextToText.from_pretrained(release['model'],revision=release['revision'],local_files_only=True,
          trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
    model=PeftModel.from_pretrained(base,run/'adapter',local_files_only=True).eval()
    output=run/'jevbench-public';output.mkdir(exist_ok=False)
    tasks=[]
    for tier in ('easy','original','hard'):
        for line in (BENCH/f'datasets/public/{tier}.jsonl').read_text().splitlines():
            task=json.loads(line);task['tier']=tier;tasks.append(task)
    summary={'scope':'231 public items only; not the 534-item private/public ranked benchmark',
        'benchmark_revision':manifest['revision'],'checkpoint_frozen_before_run':freeze,
        'max_tokens':8192,'calibration_note':'Temperature frozen on synthetic calibration_fit, transferred without refitting. Context guard raised from 1536 to 8192 for this diagnostic; no truncation.',
        'models':{}}
    started=time.monotonic()
    for label in ('base','adapter'):
        cal=json.loads((run/f'{label}-calibration.json').read_text());records=[]
        context=model.disable_adapter() if label=='base' else nullcontext()
        with context,torch.inference_mode(),(output/f'{label}.jsonl').open('w') as stream:
            for task in tasks:
                if time.monotonic()-started>1200:raise TimeoutError('Benchmark work deadline')
                item={'id':task['id'],'tier':task['tier'],'family':task['family'],'group':task.get('group'),'expected':task['expected'],'excluded':bool(task.get('provenance',{}).get('exclude_reason'))}
                t=time.perf_counter()
                try:
                    compiled=compile_request(tokenizer,to_request(task),max_tokens=8192)
                    batch=collate([compiled],tokenizer.pad_token_id,'cuda')
                    scores=candidate_logits(model,batch)[0].float().cpu().tolist()
                    p=dict(zip(task['labels'],probabilities(scores,cal['temperature'])))
                    item.update(scores=scores,probs=p,tokens=len(compiled['input_ids']),prompt_sha256=compiled['prompt_sha256'],**{k:v for k,v in score_task(p,SimpleNamespace(**task)).items() if k!='probs'})
                    gold=task.get('provenance',{}).get('gold_probs')
                    if gold:item['gold_probability_mae']=sum(abs(p[k]-gold[k]) for k in gold)/len(gold)
                except ValueError as e:
                    item.update(valid=False,correct=False if task['expected'] is not None else None,error=str(e))
                item['latency_s']=time.perf_counter()-t
                stream.write(json.dumps(item,allow_nan=False)+'\n');stream.flush();records.append(item)
                if len(records)%25==0:print(label,len(records),'/',len(tasks),flush=True)
        summaries={}
        for tier in ('easy','original','hard','all'):
            rs=[r for r in records if tier=='all' or r['tier']==tier]
            scored=[r for r in rs if r['expected'] is not None and not r['excluded']]
            pairs=[(max(r['probs'].values()),r['correct']) for r in scored if r.get('valid')]
            gold=[r['gold_probability_mae'] for r in rs if 'gold_probability_mae' in r]
            summaries[tier]={'attempts':len(rs),'scorable':len(scored),'accuracy':sum(bool(r['correct']) for r in scored)/len(scored),
                'invalid':sum(not r['valid'] for r in rs),'ece_10_bins_valid_labeled':ece_top_label(pairs)['ece'],
                'gold_probability_mae':sum(gold)/len(gold) if gold else None,'gold_distribution_items':len(gold),
                'latency':latency_summary([r['latency_s'] for r in rs])}
        summary['models'][label]=summaries
        print(json.dumps({'event':'public_benchmark_result','model':label,**summaries}),flush=True)
    summary['wall_seconds']=time.monotonic()-started
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':main()
