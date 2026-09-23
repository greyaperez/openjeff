"""Frozen OpenJeff pilot scorer. Weights must be explicitly staged beforehand."""
import json
from pathlib import Path
from ..prompting import compile_request
from ..training import adapter_digest, collate, candidate_logits
from ..scorer_identity import metadata, scorer_id

class PilotScorer:
    def __init__(self, run_path, device='cuda'):
        import torch
        from transformers import AutoTokenizer, AutoModelForImageTextToText
        from peft import PeftModel
        self.torch,self.device=torch,device
        run=Path(run_path)
        release=json.loads((run/'release.json').read_text())
        model_id,revision=release['model'],release['revision']
        self.tokenizer=AutoTokenizer.from_pretrained(model_id,revision=revision,local_files_only=True,trust_remote_code=False)
        self.model=AutoModelForImageTextToText.from_pretrained(model_id,revision=revision,local_files_only=True,
            trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='sdpa').to(device)
        adapter=None
        if release['selected_backend']=='adapter':
            adapter=adapter_digest(run/'adapter')
            if adapter!=release['package_adapter_sha256'] or adapter!=release['evaluation_adapter_sha256']:
                raise ValueError('Adapter package differs from calibrated evaluation artifact')
            self.model=PeftModel.from_pretrained(self.model,run/'adapter',local_files_only=True)
        elif release['selected_backend']!='base':raise ValueError('Unknown release backend')
        self.model.eval()
        self.max_tokens=1536
        self.scorer_id=scorer_id(metadata(model_id,revision,self.tokenizer,self.max_tokens,device),adapter)
        from ..calibration import Calibration
        self.calibration=Calibration(**json.loads((run/(release['selected_backend']+'-calibration.json')).read_text()))
        if self.calibration.scorer_id!=self.scorer_id:
            raise ValueError('Runtime differs from frozen calibrator; recalibrate this exact runtime before serving')

    def score(self,request,intermediate=None):
        if intermediate is not None:raise ValueError('This pilot is original-evidence only')
        compiled=compile_request(self.tokenizer,request,max_tokens=self.max_tokens)
        batch=collate([compiled],self.tokenizer.pad_token_id,self.device)
        with self.torch.inference_mode():
            return candidate_logits(self.model,batch)[0].float().cpu().tolist()
