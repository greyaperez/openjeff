"""Record diffusion decisions and bounded evidence guidance on every frozen input."""
import argparse,importlib.metadata,json,time
from pathlib import Path
from openjeff.calibration import digest
from openjeff.guidance import evidence_view,guidance_questions,guidance_state,make_ir
from scripts.diffusion_eval import request,extract
from scripts.hybrid_common import ROOT,SPLITS,cases,core,finish_scores

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',default='runs/hybrid-v2');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 import torch
 runtime={'model':'google/diffusiongemma-26B-A4B-it','revision':'f7f5b7f5fa82ffc52addd066915886d497f5517b','gpu':torch.cuda.get_device_name(),
  'torch':torch.__version__,'vllm':importlib.metadata.version('vllm'),'transformers':importlib.metadata.version('transformers'),
  'read_steps':1,'samples':1,'seed':0,'canvas_bound':128,'compact_canvas':True,'guidance_code_sha256':digest((ROOT/'openjeff/guidance.py').read_text()),'runtime_commit':'3ce907e6835212f27ee82b4cee9039198c4abe35'}
 identity=digest(runtime);options={'seed':0,'samples':1,'steps':1,'diagnostics':True,'score_mode':'categorical','isolation':'joint'}
 (out/'diffusion-runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
 request({'state':{'value':1},'questions':{'decision':{'type':'choice','instructions':'Choose the value','criteria':{'one':'1','two':'2'}}},'options':options})
 begin=time.monotonic()
 for split in SPLITS:
  path=out/f'diffusion-{split}.jsonl'
  if path.exists():raise FileExistsError(path)
  with path.open('w') as stream:
   for i,(row,req,native) in enumerate(cases(split)):
    if time.monotonic()-begin>4500:raise TimeoutError('Diffusion phase exceeded 75 minutes')
    start=time.perf_counter();body=request({'state':req.state,'questions':{'decision':native},'options':options});scores,probs=extract(body,[c.id for c in req.candidates]);direct_time=time.perf_counter()-start
    view=evidence_view(req);start=time.perf_counter();guided=request({'state':guidance_state(view),'questions':guidance_questions(view),'options':options})
    values={k:v['noul'] for k,v in guided['answers'].items()};ir=make_ir(view,values);guide_time=time.perf_counter()-start
    item={**core(row),'scores':scores,'scorer_id':identity,'probabilities_as_returned':probs,'ir':ir,'guidance_values':values,
      'latency_s':direct_time,'guidance_latency_s':guide_time,'candidate_ids':[c.id for c in req.candidates],'evidence_reference_ids':list(view.evidence_ids),
      'usage':body.get('usage'),'guidance_usage':guided.get('usage')}
    if 'tier' in row:item['tier']=row['tier']
    stream.write(json.dumps(item,allow_nan=False)+'\n');stream.flush()
    if (i+1)%25==0:print(json.dumps({'stage':'diffusion','split':split,'rows':i+1,'elapsed_s':round(time.monotonic()-begin,1)}),flush=True)
 summary=finish_scores(out,'diffusion',runtime);print(json.dumps({'event':'diffusion_complete','final_accuracy':summary['splits']['final']['raw']['accuracy']}),flush=True)
if __name__=='__main__':main()
