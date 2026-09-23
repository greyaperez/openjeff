"""Score frozen AR controls and diffusion/AR-guided refinement, without training."""
import argparse,json,time
from pathlib import Path
from openjeff.calibration import digest,probabilities
from openjeff.guidance import evidence_view,question_requests,make_ir
from openjeff.prompting import compile_request
from openjeff.training import adapter_digest,collate,candidate_logits
from scripts.hybrid_common import ROOT,SPLITS,cases,core,read,finish_scores

METHODS=('ar','ar_view','diffusion_guided_ar','ar_guided_ar')
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',default='runs/hybrid-v2');p.add_argument('--adapter',default='runs/pilot-v1/adapter');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 import torch,peft,transformers
 from transformers import AutoModelForImageTextToText,AutoTokenizer
 from peft import PeftModel
 model_id='google/gemma-4-12B-it';revision='707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7'
 tokenizer=AutoTokenizer.from_pretrained(model_id,revision=revision,trust_remote_code=False)
 base=AutoModelForImageTextToText.from_pretrained(model_id,revision=revision,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
 model=PeftModel.from_pretrained(base,a.adapter,local_files_only=True).eval()
 runtime={'model':model_id,'revision':revision,'adapter_sha256':adapter_digest(a.adapter),'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
  'transformers':transformers.__version__,'peft':peft.__version__,'attention':'sdpa','max_tokens':8192,'guidance_code_sha256':digest((ROOT/'openjeff/guidance.py').read_text()),'prompt':'codes-v1','self_guidance_batch_size':8}
 ids={method:digest({**runtime,'method':method}) for method in METHODS}
 (out/'ar-runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
 def score(req,ir=None):
  start=time.perf_counter();compiled=compile_request(tokenizer,req,max_tokens=8192,intermediate=ir)
  with torch.inference_mode():values=candidate_logits(model,collate([compiled],tokenizer.pad_token_id,'cuda'))[0].float().cpu().tolist()
  return values,time.perf_counter()-start,len(compiled['input_ids'])
 def self_guidance(view):
  start=time.perf_counter();requests=question_requests(view);keys=list(requests);values={}
  for j in range(0,len(keys),8):
   ks=keys[j:j+8];compiled=[compile_request(tokenizer,requests[k],max_tokens=8192) for k in ks]
   with torch.inference_mode():batch=candidate_logits(model,collate(compiled,tokenizer.pad_token_id,'cuda')).float().cpu().tolist()
   values.update({k:probabilities(s)[1] for k,s in zip(ks,batch)})
  return make_ir(view,values),values,time.perf_counter()-start
 # Warm model before timing; source request only, never its label.
 _,warm,_=next(cases('development'));score(warm)
 begin=time.monotonic()
 for split in SPLITS:
  guidance={r['id']:r for r in read(out/f'diffusion-{split}.jsonl')}
  paths={m:out/f'{m}-{split}.jsonl' for m in METHODS}
  if any(path.exists() for path in paths.values()):raise FileExistsError('Refusing to overwrite scored arms')
  streams={m:path.open('w') for m,path in paths.items()}
  try:
   for i,(row,req,_) in enumerate(cases(split)):
    if time.monotonic()-begin>5400:raise TimeoutError('AR phase exceeded 90 minutes')
    view=evidence_view(req);guide=guidance[row['id']]
    assert guide['input_sha256']==row['input_sha256'] and guide['target']==row['target']
    self_ir,self_values,self_time=self_guidance(view)
    for method,input_request,ir,first_time in [('ar',req,None,0),('ar_view',view,None,0),('diffusion_guided_ar',view,guide['ir'],guide['guidance_latency_s']),('ar_guided_ar',view,self_ir,self_time)]:
     scores,elapsed,tokens=score(input_request,ir)
     record={**core(row),'scorer_id':ids[method],'scores':scores,'latency_s':elapsed,'first_stage_latency_s':first_time,'serial_pipeline_latency_s':elapsed+first_time,'tokens':tokens}
     if 'tier' in row:record['tier']=row['tier']
     if ir is not None:record['ir']=ir
     if method=='ar_guided_ar':record['guidance_values']=self_values
     streams[method].write(json.dumps(record,allow_nan=False)+'\n');streams[method].flush()
    if (i+1)%25==0:print(json.dumps({'stage':'ar','split':split,'rows':i+1,'elapsed_s':round(time.monotonic()-begin,1)}),flush=True)
  finally:
   for stream in streams.values():stream.close()
 for method in METHODS:
  result=finish_scores(out,method,{**runtime,'method':method});print(json.dumps({'event':'ar_arm_complete','method':method,'final_accuracy':result['splits']['final']['raw']['accuracy']}),flush=True)
if __name__=='__main__':main()
