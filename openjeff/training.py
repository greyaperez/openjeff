"""Candidate-only LoRA helpers, shared by training and batched inference."""
import hashlib
from pathlib import Path


def adapter_digest(path):
    checksum = hashlib.sha256()
    for name in ('adapter_config.json', 'adapter_model.safetensors'):
        data = (Path(path) / name).read_bytes()
        checksum.update(name.encode() + b'\0' + data)
    return checksum.hexdigest()


def attach_lora(model, rank=16, alpha=32, dropout=0.05):
    from peft import LoraConfig, get_peft_model
    # Exact module names prevent accidentally adapting the vision/audio towers.
    targets = [name for name, module in model.named_modules()
               if ('.language_model.layers.' in name or name.startswith('model.layers.'))
               and name.rsplit('.', 1)[-1] in {'q_proj', 'k_proj', 'v_proj', 'o_proj'}
               and module.__class__.__name__ == 'Linear']
    if not targets:
        raise ValueError('No supported text attention projections found')
    return get_peft_model(model, LoraConfig(r=rank, lora_alpha=alpha,
        lora_dropout=dropout, target_modules=targets, bias='none'))


def collate(examples, pad_id, device):
    import torch
    if not examples:
        raise ValueError('Empty batch')
    length = max(len(x['input_ids']) for x in examples)
    choices = max(len(x['code_ids']) for x in examples)
    if any(not x['input_ids'] or len(x['code_ids']) < 2 for x in examples):
        raise ValueError('Invalid compiled example')
    inputs = torch.full((len(examples), length), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(inputs)
    codes = torch.zeros((len(examples), choices), dtype=torch.long, device=device)
    valid = torch.zeros_like(codes, dtype=torch.bool)
    for i, row in enumerate(examples):
        n, k = len(row['input_ids']), len(row['code_ids'])
        inputs[i, -n:] = torch.tensor(row['input_ids'], device=device)
        mask[i, -n:] = 1
        codes[i, :k] = torch.tensor(row['code_ids'], device=device)
        valid[i, :k] = True
    return {'input_ids': inputs, 'attention_mask': mask,
            'position_ids': (mask.cumsum(-1) - 1).clamp(min=0),
            'codes': codes, 'valid': valid}


def candidate_logits(model, batch):
    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                    position_ids=batch['position_ids'], use_cache=False, logits_to_keep=1)
    return outputs.logits[:, -1].float().gather(1, batch['codes']).masked_fill(~batch['valid'], float('-inf'))


def candidate_loss(logits, targets, valid):
    import torch
    if targets.dtype != torch.long or targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError('One integer target per row required')
    if not ((targets >= 0) & (targets < logits.shape[1])).all() or not valid.gather(1, targets[:, None]).all():
        raise ValueError('Target points to an absent candidate')
    return torch.nn.functional.cross_entropy(logits, targets)
