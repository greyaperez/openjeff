"""Atomic vision-prefix attention and admission for up to six image attachments."""
from __future__ import annotations

MAX_IMAGES = 6


def enabled(model_config):
    return getattr(getattr(model_config, 'hf_config', None), 'model_type', None) == 'diffusion_gemma'


def validate_configuration(config, features):
    """Reject unsupported image execution modes before an EngineCoreRequest exists."""
    if not enabled(config.model_config) or not any(item.modality == 'image' for item in (features or ())):
        return
    parallel = config.parallel_config
    if any(getattr(parallel, name, None) != 1 for name in (
        'tensor_parallel_size', 'pipeline_parallel_size', 'data_parallel_size',
        'decode_context_parallel_size', 'prefill_context_parallel_size')):
        raise ValueError('Vision attention supports one worker without distributed parallelism')
    if getattr(parallel, 'use_ubatching', True):
        raise ValueError('Vision attention does not support microbatching')
    if (getattr(config, 'num_prefill_lookahead_tokens', None) != 0
            or getattr(config, 'speculative_config', None) is not None):
        raise ValueError('Vision attention does not support speculative prefill')
    if (getattr(config, 'kv_transfer_config', None) is not None
            or getattr(config, 'ec_transfer_config', None) is not None):
        raise ValueError('Vision attention does not support external cache connectors')


def image_spans(model_config, features, prompt_length):
    if not enabled(model_config) or not features:
        return None
    images = [item for item in features if item.modality == 'image']
    if not images:
        return None
    if not 1 <= len(images) <= MAX_IMAGES or len(features) != len(images):
        raise ValueError('Vision attention requires between one and six image spans')
    text = model_config.hf_text_config
    if getattr(text, 'use_bidirectional_attention', None) != 'vision':
        raise ValueError('Vision attention requires the pinned vision configuration')
    spans = []
    window = getattr(text, 'sliding_window', None)
    if type(window) is not int or window < 1:
        raise ValueError('Vision attention requires the pinned sliding window')
    for image in images:
        ranges = list(image.mm_position.extract_embeds_range())
        if (len(ranges) != 1 or len(ranges[0]) != 2
                or any(type(value) is not int for value in ranges[0])):
            raise ValueError('Each image requires one valid embedding span')
        begin, end = ranges[0]
        if not 0 <= begin <= end < prompt_length:
            raise ValueError('Vision embedding span exceeds the rendered prompt')
        if end-begin+1 > min(window, 1024):
            raise ValueError('Vision embedding span exceeds the supported sliding window')
        spans.append((begin, end))
    # Feature iteration order is not an attention-coordinate contract. Sorting
    # also lets scheduling stop before the earliest partial image, including
    # when one chunk contains several complete image blocks.
    spans.sort()
    if any(previous[1] >= current[0] for previous, current in zip(spans, spans[1:])):
        raise ValueError('Vision image spans must not overlap')
    return spans



def admit(model_config, scheduler_config, features, sampling_params,
          prompt_length, default_canvas):
    """Called on cloned parameters and processed placeholders before core dispatch."""
    spans = image_spans(model_config, features, prompt_length)
    if spans is None:
        return
    extra = getattr(sampling_params, 'extra_args', None) or {}
    if extra.get('diffusion_read_only') not in (True, 1) or extra.get('diffusion_max_steps') != 1:
        raise ValueError('Vision attention supports one-step read-only requests')
    canvas = extra.get('diffusion_canvas_length', default_canvas)
    if type(canvas) is not int or not 1 <= canvas <= 128:
        raise ValueError('Vision attention requires a bounded canvas')
    limit = model_config.max_model_len
    if type(limit) is not int or limit > 32768 or prompt_length+canvas > limit:
        raise ValueError('Vision prompt plus reserved canvas exceeds the context limit')
    budget = scheduler_config.max_num_batched_tokens
    threshold = getattr(scheduler_config, 'long_prefill_token_threshold', 0)
    if threshold > 0:
        budget = min(budget, threshold)
    if any(budget < end-begin+1 for begin, end in spans):
        raise ValueError('Vision image span exceeds the configured prefill budget')
    # Retain encoder/processor reuse. Only text-backbone KV prefix reuse is
    # bypassed; those three caches have different correctness conditions.
    sampling_params.skip_reading_prefix_cache = True


def atomic_image_budget(model_config, request, start, count, shift=0):
    """Run before feature-window filtering or encoder-cache early continues."""
    features = getattr(request, 'mm_features', None)
    if not enabled(model_config) or not features:
        return count
    spans = image_spans(model_config, features, request.num_prompt_tokens)
    if spans is None:
        return count
    if shift:
        raise ValueError('Vision attention does not support shifted speculative prefill')
    finish = start+count
    for begin, end in spans:
        if begin < start <= end:
            raise ValueError('Vision prefill cannot restart inside an image span')
        if start <= begin < finish <= end:
            # Earlier full images and intervening text remain scheduled. The
            # next attempt begins before this image, never inside its block.
            return begin-start
    return count



def ranges_for_batch(state, batch, for_capture=False):
    """Only image-query rows receive an overlay; indices are actual batch order."""
    if batch is None or not enabled(state.model_config):
        return None
    features = getattr(state.encoder_cache, 'mm_features', {})
    result = {}
    for index, request_id in enumerate(batch.req_ids):
        if not bool(batch.is_prefilling_np[index]):
            continue
        spans = image_spans(state.model_config, features.get(request_id),
                            int(batch.prefill_len_np[index]))
        if spans is None:
            continue
        start = int(batch.num_computed_prefill_tokens_np[index])
        finish = start+int(batch.num_scheduled_tokens[index])
        active = []
        for begin, end in spans:
            if finish <= begin or start > end:
                continue
            if start > begin or finish <= end:
                raise ValueError('Vision attention cannot process a partial image span')
            active.append((begin, end))
        if active:
            if for_capture:
                raise ValueError('Vision image-prefill attention cannot use graph capture')
            result[index] = active
    return result or None


def needs_eager(state, batch):
    # Uses actual registered features, including vision-encoder cache hits.
    # One image row makes the entire mixed batch eager; no per-row graph mode.
    return bool(ranges_for_batch(state, batch))
