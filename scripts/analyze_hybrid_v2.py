"""Frozen development selection, independent calibration, and paired reporting."""
import argparse,collections,json,math,random,statistics
from pathlib import Path
from openjeff.calibration import digest,probabilities,mean_nll
from openjeff.evaluation import evaluate,refinement_metrics
from scripts.hybrid_common import ROOT,SPLITS,read,finish_scores
from scripts.audit_curriculum_overlap import state_key

METHODS=('ar','ar_view','diffusion','diffusion_guided_ar','ar_guided_ar','fusion')

def aligned(a,b):
    if len(a)!=len(b):raise ValueError('Unequal paired sample sizes')
    for x,y in zip(a,b):
        for key in ('id','target','input_sha256'):
            if x[key]!=y[key]:raise ValueError('Paired input mismatch: '+key)
        if len(x['scores'])!=len(y['scores']):raise ValueError('Candidate size mismatch')

def mix(a,b,weight):
    aligned(a,b);rows=[]
    identity=digest({'method':'raw_probability_mixture','weight_diffusion':weight,'ar':a[0]['scorer_id'],'diffusion':b[0]['scorer_id']})
    for x,y in zip(a,b):
        ps=[(1-weight)*p+weight*q for p,q in zip(probabilities(x['scores']),probabilities(y['scores']))]
        latency=x['latency_s'] if weight==0 else y['latency_s'] if weight==1 else x['latency_s']+y['latency_s']
        rows.append({**x,'scores':[math.log(max(p,1e-300)) for p in ps],'scorer_id':identity,'latency_s':latency,'serial_pipeline_latency_s':latency,'latency_scope':'required warmed stages; endpoint weight omits the unused backend'})
    return rows

def correct(rows):return [max(range(len(r['scores'])),key=r['scores'].__getitem__)==r['target'] for r in rows]
def compact(metrics):return {k:v for k,v in metrics.items() if k not in ('risk_coverage','reliability')}
def latency(rows):
    values=sorted(r.get('serial_pipeline_latency_s',r['latency_s']) for r in rows)
    return {'mean_s':statistics.mean(values),'p50_s':statistics.median(values),'p95_s':values[math.ceil(.95*len(values))-1],'scope':'serial observed service/model time; excludes download, model startup, GPU swap and queueing at concurrency'}

def paired(a,b):
    aligned(a,b);aa,bb=correct(a),correct(b);stats=refinement_metrics(aa,bb)
    stats.update(repaired=sum(not x and y for x,y in zip(aa,bb)),broken=sum(x and not y for x,y in zip(aa,bb)))
    # Shared templates violate independent-row bootstrap assumptions. Resample
    # task families as clusters; this small-family interval remains descriptive.
    groups=collections.defaultdict(list)
    for r,x,y in zip(a,aa,bb):groups[r['family']].append(int(y)-int(x))
    keys=list(groups);rng=random.Random(23);samples=[]
    for _ in range(2000):
        values=[d for k in rng.choices(keys,k=len(keys)) for d in groups[k]]
        samples.append(sum(values)/len(values))
    samples.sort();stats['descriptive_family_cluster_interval']={'lower':samples[50],'upper':samples[1949],'families':len(keys),'resamples':2000,'seed':23,'warning':'Not a guarantee for unseen domains; small, shared generator families.'}
    return stats

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',default='runs/hybrid-v2');args=parser.parse_args();out=Path(args.out)
    a,b=read(out/'ar-development.jsonl'),read(out/'diffusion-development.jsonl')
    candidates=[{'weight_diffusion':w,'development_nll':mean_nll(mix(a,b,w),1)} for w in (0,.25,.5,.75,1)]
    weight=min(candidates,key=lambda r:r['development_nll'])['weight_diffusion']
    (out/'fusion-selection.json').write_text(json.dumps({'selection_split':'development','metric':'raw NLL','weights':candidates,'selected_weight_diffusion':weight},indent=2)+'\n')
    for split in SPLITS:
        rows=mix(read(out/f'ar-{split}.jsonl'),read(out/f'diffusion-{split}.jsonl'),weight)
        (out/f'fusion-{split}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
    finish_scores(out,'fusion',{'method':'raw_probability_mixture','weight_diffusion':weight})
    all_rows={m:{s:read(out/f'{m}-{s}.jsonl') for s in SPLITS} for m in METHODS}
    temps={m:json.loads((out/f'{m}-calibration.json').read_text())['temperature'] for m in METHODS}
    train_states={state_key(r) for r in read(ROOT/'data/curriculum-v1/train.jsonl')}
    nonmatching={r['id'] for r in read(ROOT/'data/curriculum-v1/final.jsonl') if state_key(r) not in train_states}
    result={'scope':'Inference-only structured diffusion guidance, no trained diffusion adapter or latent bridge. Frozen AR weights.','selected_weight_diffusion':weight,'methods':{},'paired':{}}
    public_tasks={r['id']:r for tier in ('easy','original','hard') for r in read(ROOT/f'research/snapshots/jevbench/datasets/public/{tier}.jsonl')}
    for m in METHODS:
        result['methods'][m]={'temperature':temps[m],'splits':{}}
        subsets={s:all_rows[m][s] for s in ('calibration_check','final','adversarial','production_sim','public')}
        subsets['final_no_normalized_train_match']=[r for r in all_rows[m]['final'] if r['id'] in nonmatching]
        subsets.update({f'public_{tier}':[r for r in all_rows[m]['public'] if r['tier']==tier] for tier in ('easy','original','hard')})
        for name,rows in subsets.items():
            bins=10 if name.startswith('public') else 15
            result['methods'][m]['splits'][name]={'raw':compact(evaluate(rows,bins=bins)),'calibrated':compact(evaluate(rows,temps[m],bins=bins)),'serial_latency':latency(rows)}
            if name.startswith('public'):
                gold_errors=[]
                for r in rows:
                    task=public_tasks[r['id']];gold=task.get('provenance',{}).get('gold_probs')
                    if gold:
                        probs=dict(zip(task['labels'],probabilities(r['scores'],temps[m])))
                        gold_errors.append(sum(abs(probs[k]-gold[k]) for k in gold)/len(gold))
                result['methods'][m]['splits'][name]['soft_gold_probability_mae']={'n':len(gold_errors),'mean':statistics.mean(gold_errors) if gold_errors else None}
    for name in ('final','adversarial','production_sim','public'):
        result['paired'][name]={}
        for first,last in [('ar','diffusion'),('ar','ar_view'),('ar','diffusion_guided_ar'),('ar_view','diffusion_guided_ar'),('ar_guided_ar','diffusion_guided_ar'),('diffusion','diffusion_guided_ar'),('ar','fusion')]:
            result['paired'][name][first+' -> '+last]=paired(all_rows[first][name],all_rows[last][name])
    (ROOT/'reports/hybrid-v2-results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'selected_weight_diffusion':weight,'final_accuracy':{m:result['methods'][m]['splits']['final']['raw']['accuracy'] for m in METHODS},'public_accuracy':{m:result['methods'][m]['splits']['public']['raw']['accuracy'] for m in METHODS}},indent=2))

if __name__=='__main__':main()
