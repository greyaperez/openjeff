"""Reference local HF scorer. Gemma 4 12B/E2B BF16 paths probed on H100.

Install a compatible, pinned Transformers/PyTorch environment separately.
The model/tokenizer must already be in the local HF cache at an immutable SHA.
"""
import re
from ..calibration import digest
from ..prompting import compile_request, PROMPT_VERSION


class HFLabelScorer:
    def __init__(self, model_id, revision, *, max_tokens=8192, device="cuda", mode="original"):
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("immutable 40-character revision required")
        if mode not in {"original", "structured_ir"}:
            raise ValueError("unknown scorer mode")
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        import torch
        import transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision,
                                                       local_files_only=True, trust_remote_code=False)
        loader = AutoModelForImageTextToText if model_id.startswith("google/gemma-4-") else AutoModelForCausalLM
        self.model = loader.from_pretrained(model_id, revision=revision, local_files_only=True,
                                           trust_remote_code=False, dtype=torch.bfloat16).to(device).eval()
        self.max_tokens, self.device, self.mode = max_tokens, device, mode
        self.scorer_id = digest({"model": model_id, "revision": revision, "prompt": PROMPT_VERSION,
                                 "torch": torch.__version__, "transformers": transformers.__version__,
                                 "max_tokens": max_tokens, "dtype": "bf16", "mode": mode,
                                 "device": device, "chat_template": self.tokenizer.chat_template})

    def score(self, request, intermediate=None):
        if (intermediate is not None) != (self.mode == "structured_ir"):
            raise ValueError("IR mode must match the frozen scorer configuration")
        compiled = compile_request(self.tokenizer, request, max_tokens=self.max_tokens, intermediate=intermediate)
        prefix, code_ids = compiled["input_ids"], compiled["code_ids"]
        ids = self.torch.tensor([prefix], device=self.device)
        with self.torch.inference_mode():
            # Fail if the selected backend lacks this API; never silently allocate
            # all prompt-position vocabulary logits as a fallback.
            outputs = self.model(input_ids=ids, attention_mask=self.torch.ones_like(ids),
                                 use_cache=False, logits_to_keep=1)
        return outputs.logits[0, -1, code_ids].float().cpu().tolist()
