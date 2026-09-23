"""Explicit public weight download at the approved immutable revision (about 24 GB)."""
import argparse,json
from pathlib import Path
from scripts.gpu_probe import model_entry

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--model',default='google/gemma-4-12B-it',choices=['google/gemma-4-12B-it']);args=parser.parse_args()
    from huggingface_hub import snapshot_download
    entry=model_entry(args.model)
    path=snapshot_download(args.model,revision=entry['revision'],token=False,
        allow_patterns=['*.safetensors','*.safetensors.index.json','config.json','generation_config.json','tokenizer*','special_tokens_map.json','chat_template.jinja','LICENSE*','NOTICE*','README.md'])
    print(json.dumps({'model':args.model,'revision':entry['revision'],'local_snapshot':path,'inference_performed':False},indent=2))

if __name__=='__main__':main()
