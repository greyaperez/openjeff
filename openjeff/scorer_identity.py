"""Frozen inference identity used for calibration compatibility checks."""
from .calibration import digest
from .prompting import PROMPT_VERSION

def metadata(model, revision, tokenizer, max_tokens, device):
    import torch, transformers, peft
    return {'model':model,'revision':revision,'prompt':PROMPT_VERSION,'max_tokens':max_tokens,
        'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,
        'gpu':torch.cuda.get_device_name(device) if str(device).startswith('cuda') else 'cpu',
        'dtype':'bf16','attention':'sdpa','chat_template_sha256':digest(tokenizer.chat_template),
        'evaluation_batch_size':1}

def scorer_id(meta, adapter_sha256=None):
    return digest(meta | {'adapter_sha256':adapter_sha256})
