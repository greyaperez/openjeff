"""Bounded, reproducible BF16 candidate-loss LoRA pilot; never provisions resources."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from openjeff.calibration import digest, fit_temperature
from openjeff.evaluation import evaluate, paired_accuracy_interval, refinement_metrics
from openjeff.probe_cases import to_request
from openjeff.prompting import compile_request, PROMPT_VERSION
from openjeff.training import attach_lora, adapter_digest, collate, candidate_logits, candidate_loss
from scripts.gpu_probe import model_entry

MODEL = 'google/gemma-4-12B-it'

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def compact(metrics):
    return {k:v for k,v in metrics.items() if k not in ('reliability','risk_coverage')}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='runs/pilot-v1')
    parser.add_argument('--data', default='data/curriculum-v1')
    parser.add_argument('--microbatch', type=int, default=4)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--max-seconds', type=int, default=4500)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.steps <= 500 or not 60 <= args.max_seconds <= 5400 or not 1 <= args.microbatch <= 8 or not 1 <= args.accumulation <= 16:
        parser.error('Unsafe work bounds')
    import torch
    import transformers
    import peft
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    from huggingface_hub import snapshot_download
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required before downloads')
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    torch.manual_seed(230923); torch.cuda.manual_seed_all(230923); random.seed(230923)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.monotonic()
    def deadline():
        if time.monotonic() - started > args.max_seconds:
            raise TimeoutError('Work deadline reached. This stops computation, not provider billing.')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    data = Path(args.data)
    manifest = json.loads((data/'manifest.json').read_text())
    for entry in manifest['files']:
        if hashlib.sha256((data/entry['path']).read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('Dataset checksum mismatch')
    revision = model_entry(MODEL)['revision']
    snapshot = Path(snapshot_download(MODEL, revision=revision, token=False, local_files_only=not args.download,
        allow_patterns=['*.safetensors', '*.safetensors.index.json', 'config.json', 'generation_config.json',
                        'tokenizer*', 'special_tokens_map.json', 'chat_template.jinja', 'LICENSE*', 'NOTICE*', 'README.md']))
    base_files = []
    for p in sorted(snapshot.iterdir()):
        if p.is_file():
            h = hashlib.sha256()
            with p.open('rb') as stream:
                for block in iter(lambda:stream.read(8*1024*1024), b''): h.update(block)
            base_files.append({'name':p.name,'bytes':p.stat().st_size,'sha256':h.hexdigest()})
    write(out/'base-files.json', {'model':MODEL,'revision':revision,'files':base_files})
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    def load_split(split):
        rows = [json.loads(line) for line in (data/(split+'.jsonl')).read_text().splitlines()]
        for row in rows:
            if row['split'] != split: raise ValueError('Split mismatch')
            row['compiled'] = compile_request(tokenizer, to_request(row), max_tokens=1536)
        return rows
    train, development = load_split('train'), load_split('development')
    model = AutoModelForImageTextToText.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False,
                dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    model = attach_lora(model)
    ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad)
    runtime = {'model':MODEL,'revision':revision,'prompt':PROMPT_VERSION,'max_tokens':1536,
        'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,
        'gpu':torch.cuda.get_device_name(),'dtype':'bf16','attention':'sdpa',
        'chat_template_sha256':digest(tokenizer.chat_template), 'evaluation_batch_size':1,
        'training':dict(vars(args),seed=230923,lr=0.0001,rank=16,alpha=32,dropout=.05,
                        trainable_parameters=ntrain,gradient_checkpointing=False),
        'dataset_manifest':manifest,'scope':'synthetic rule tasks; not JevBench'}
    write(out/'runtime.json', runtime)
    def identity(adapter):
        return digest({k:runtime[k] for k in ('model','revision','prompt','max_tokens','torch','transformers',
            'peft','gpu','dtype','attention','chat_template_sha256','evaluation_batch_size')} | {'adapter_sha256':adapter})
    def score(rows, filename, adapter, disabled=False, reverse=False):
        deadline(); model.eval(); scored=[]
        context = model.disable_adapter() if disabled else nullcontext()
        with context, torch.inference_mode(), (out/filename).open('w') as stream:
            for row in rows:
                deadline()
                if reverse:
                    from copy import deepcopy
                    row = deepcopy(row)
                    row['request']['candidates'].reverse()
                    row['target'] = len(row['request']['candidates']) - 1 - row['target']
                    row['input_sha256'] = digest(row['request'])
                    row['compiled'] = compile_request(tokenizer,to_request(row),max_tokens=1536)
                batch = collate([row['compiled']], tokenizer.pad_token_id, 'cuda')
                torch.cuda.synchronize(); t=time.perf_counter()
                logits = candidate_logits(model,batch)[0].cpu().tolist()
                torch.cuda.synchronize(); elapsed=time.perf_counter()-t
                item = {k:row[k] for k in ('id','group_id','split','family','input_sha256','target')}
                item.update(scores=logits,scorer_id=identity(adapter),latency_s=elapsed,
                            candidate_ids=[x['id'] for x in row['request']['candidates']])
                stream.write(json.dumps(item,allow_nan=False)+'\n'); scored.append(item)
        print(json.dumps({'event':'evaluation','file':filename,**compact(evaluate(scored))}),flush=True)
        return scored
    baseline_dev = score(development,'base-development.jsonl',None,disabled=True)
    best_key = (-1.0,float('-inf')); best_step=None
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
    order = list(range(len(train))); random.Random(230923).shuffle(order); cursor=0
    milestones = sorted({min(125,args.steps),min(250,args.steps),args.steps})
    log = (out/'training.jsonl').open('w')
    torch.cuda.reset_peak_memory_stats(); model.train(); optimizer.zero_grad(set_to_none=True)
    for step in range(1,args.steps+1):
        deadline(); loss_sum=0; t=time.monotonic()
        lr=1e-4 * min(1., step/25) * (.1 + .9*.5*(1+math.cos(math.pi*step/args.steps)))
        for group in optimizer.param_groups: group['lr']=lr
        for micro in range(args.accumulation):
            deadline()
            indices = [order[(cursor+j)%len(order)] for j in range(args.microbatch)]
            cursor += args.microbatch
            examples = [train[i] for i in indices]
            batch = collate([x['compiled'] for x in examples],tokenizer.pad_token_id,'cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits = candidate_logits(model,batch)
                loss = candidate_loss(logits,torch.tensor([x['target'] for x in examples],device='cuda'),batch['valid'])
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            (loss/args.accumulation).backward(); loss_sum+=loss.item()/args.accumulation
        grad = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
        if not torch.isfinite(grad): raise RuntimeError('Nonfinite gradient')
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        record={'step':step,'examples_seen':cursor,'loss':loss_sum,'lr':lr,'gradient_norm':float(grad),
                'seconds':time.monotonic()-t,'peak_memory_bytes':torch.cuda.max_memory_allocated()}
        log.write(json.dumps(record)+'\n');log.flush()
        if step==1 or step%10==0: print(json.dumps({'event':'train',**record}),flush=True)
        if step in milestones:
            dev = score(development,f'adapter-development-step{step}.jsonl',f'in-memory-step-{step}')
            metrics=evaluate(dev); key=(metrics['accuracy'],-metrics['nll'])
            if key > best_key:
                best_key, best_step=key,step
                model.save_pretrained(out/'adapter',safe_serialization=True)
                cfgpath=out/'adapter/adapter_config.json'
                cfg=json.loads(cfgpath.read_text());cfg['base_model_name_or_path']=MODEL;cfg['revision']=revision;write(cfgpath,cfg)
                write(out/'checkpoint-selection.json',{'selected_step':step,'dev_metrics':compact(metrics),
                    'rule':'highest development accuracy; lower NLL breaks ties','final_unopened':True})
            model.train()
    log.close(); del optimizer
    # Freeze selection before reading any calibration or final request.
    best_hash=adapter_digest(out/'adapter')
    base_metrics=evaluate(baseline_dev)
    release_choice='adapter' if best_key > (base_metrics['accuracy'],-base_metrics['nll']) else 'base'
    write(out/'freeze.json',{'adapter_sha256':best_hash,'best_step':best_step,'release_choice':release_choice,
        'rule':'development accuracy then NLL against base','final_unopened':True,'frozen_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    # Load the selected saved checkpoint, not the last training state.
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    set_peft_model_state_dict(model,load_file(str(out/'adapter/adapter_model.safetensors')))
    model.eval(); summary={'runtime':runtime,'release_choice':release_choice,'best_step':best_step,'models':{}}
    results={}
    for label,adapter,disabled in [('base',None,True),('adapter',best_hash,False)]:
        fit=score(load_split('calibration_fit'),f'{label}-calibration_fit.jsonl',adapter,disabled)
        cal=fit_temperature(fit);write(out/f'{label}-calibration.json',cal.to_dict())
        summary['models'][label]={'temperature':cal.temperature,'scorer_id':identity(adapter),'splits':{}}
        for split in ('calibration_check','final','adversarial','production_sim'):
            rows=load_split(split)
            scored=score(rows,f'{label}-{split}.jsonl',adapter,disabled);cal.validate_evaluation(scored)
            results[(label,split)]=scored
            metrics={'raw':evaluate(scored),'calibrated':evaluate(scored,cal.temperature),
                     'families':{f:compact(evaluate([r for r in scored if r['family']==f],cal.temperature)) for f in sorted({r['family'] for r in scored})}}
            summary['models'][label]['splits'][split]=metrics
            if split=='final':
                rev=score(rows,f'{label}-final-reversed.jsonl',adapter,disabled,reverse=True)
                winners=lambda rs:[r['candidate_ids'][max(range(len(r['scores'])),key=r['scores'].__getitem__)] for r in rs]
                summary['models'][label]['final_order_disagreement']=sum(a!=b for a,b in zip(winners(scored),winners(rev)))/len(scored)
    def correct(rows):return [max(range(len(r['scores'])),key=r['scores'].__getitem__)==r['target'] for r in rows]
    a,b=correct(results['base','final']),correct(results['adapter','final'])
    summary['paired_final']={'interval':paired_accuracy_interval(a,b),'refinement':refinement_metrics(a,b)}
    summary['wall_seconds']=time.monotonic()-started
    summary['peak_memory_bytes']=torch.cuda.max_memory_allocated()
    write(out/'summary.json',summary)
    write(out/'release.json',{'model':MODEL,'revision':revision,'selected_backend':release_choice,
        'evaluation_adapter_sha256':best_hash,'package_adapter_sha256':adapter_digest(out/'adapter'),
        'adapter_weights_sha256':hashlib.sha256((out/'adapter/adapter_model.safetensors').read_bytes()).hexdigest(),
        'scope':'experimental synthetic structured decisions','cloud_terminated':False})
    print(json.dumps({'event':'complete','release_choice':release_choice,'wall_seconds':summary['wall_seconds']}),flush=True)

if __name__=='__main__': main()
