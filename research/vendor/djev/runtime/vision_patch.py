"""Hash-guarded native vision attention source transformations."""
from __future__ import annotations

import hashlib

PINNED = {
    'model_executor/models/diffusion_gemma.py': '43f241f2f75014276eab26eb03eb52432f47ff0fa89cd36916807686d1985a89',
    'v1/engine/input_processor.py': '01177aa1dd08315034b39f580d2657fe371fd9c83d15404dca6121631dd3245c',
    'v1/core/sched/scheduler.py': 'e10fa2cba6e39b145c1e409d1d268425137c936ba69524a03433a987a9891db5',
    'v1/worker/gpu/model_runner.py': '2d9521e809b7913d6781ec62805890a43f7f3d2737acaee05e3194e5b3ff6b68',
}
IMPORT = 'from vllm.model_executor.models import djev_vision_serving as _djev_vision\n'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('Pinned vLLM source anchor differs')
    return text.replace(before, after)


def patch_source(relative, raw):
    if relative not in PINNED or sha(raw) != PINNED[relative]:
        raise ValueError('Pinned vLLM source hash differs')
    text = raw.decode()
    if relative.endswith('diffusion_gemma.py'):
        anchor = 'from vllm.v1.worker.gpu.attn_utils import build_attn_metadata\n'
        text = replace_once(text, anchor, anchor+IMPORT)
        start = text.index('    def prepare_attn(\n', text.index('class DiffusionGemmaModelState'))
        end = text.index('\n    num_new_sampled_tokens_per_step:',start)
        method = text[start:end]
        method = replace_once(method, '        return build_attn_metadata(\n',
            '        _djev_ranges = _djev_vision.ranges_for_batch(self, input_batch, for_capture)\n'
            '        if _djev_ranges and cudagraph_mode != CUDAGraphMode.NONE:\n'
            '            raise ValueError("Vision image prefill must execute eagerly")\n'
            '        return build_attn_metadata(\n')
        method = replace_once(method, '            causal=causal,\n',
            '            causal=causal,\n            mm_req_doc_ranges=_djev_ranges,\n')
        text = text[:start]+method+text[end:]
    elif relative.endswith('input_processor.py'):
        text = replace_once(text, 'from vllm.exceptions import VLLMValidationError\n',
                            'from vllm.exceptions import VLLMValidationError\n'+IMPORT)
        anchor = '        return EngineCoreRequest(\n'
        text = replace_once(text, anchor,
            '        try:\n'
            '            _djev_vision.validate_configuration(self.vllm_config, mm_features)\n'
            '            _djev_vision.admit(self.model_config, self.scheduler_config, mm_features,\n'
            '                sampling_params, len(prompt_token_ids or []),\n'
            '                getattr(self.vllm_config.diffusion_config, "canvas_length", None))\n'
            '        except ValueError as exc:\n'
            '            raise VLLMValidationError(str(exc), parameter="multi_modal_data") from exc\n\n'+anchor)
    elif relative.endswith('scheduler.py'):
        anchor = 'from vllm.config import KVEventsConfig, VllmConfig\n'
        text = replace_once(text,anchor,anchor+IMPORT)
        anchor = '        if num_new_tokens == 0 or not request.has_encoder_inputs:\n'
        text = replace_once(text,anchor,
            '        num_new_tokens = _djev_vision.atomic_image_budget(\n'
            '            self.vllm_config.model_config, request, num_computed_tokens,\n'
            '            num_new_tokens, shift_computed_tokens)\n'+anchor)
    else:
        anchor = 'from vllm.config.compilation import CUDAGraphMode\n'
        text = replace_once(text,anchor,anchor+IMPORT)
        anchor = '        skip_compiled = False\n'
        text = replace_once(text,anchor,
            '        skip_compiled = _djev_vision.needs_eager(self.model_state, batch_req_state)\n')
    compile(text, relative, 'exec')
    return text.encode()
