"""Djev image reads: true multimodal chat, bounded expanded-token preflight."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
from typing import Any

import httpx

from .contracts import MAX_IMAGE_ATTACHMENTS, description_images, question_descriptions, request_image_count
from .engine import (
    BackendError, CompiledSchema, DiffusionEngine, EngineResult, ImageState, SchemaError,
    _description, _messages,
)
from .images import validate_image_data_url

# The pinned checkpoint's image-placeholder token. The backend expands
# this through Gemma4Processor and merges vision embeddings in the model runner.
IMAGE_TOKEN_ID = 258880


@dataclass(frozen=True)
class ImageSchema(CompiledSchema):
    expanded_prompt_tokens: int


class DjevEngine(DiffusionEngine):
    supports_images = True
    def __init__(self, *args, image_preflight_concurrency: int = 2, **kwargs):
        if type(image_preflight_concurrency) is not int or not 1 <= image_preflight_concurrency <= 2:
            raise ValueError('image preflight concurrency must be 1 or 2')
        super().__init__(*args, **kwargs)
        self._image_preflight_concurrency = image_preflight_concurrency
        self._image_preflight_slots = asyncio.Semaphore(2)

    def _request_state(self, request):
        images = getattr(request, 'images', None)
        if images and (not isinstance(images, (list, tuple)) or len(images) != 1):
            raise SchemaError('at most one state image is supported per request')
        # Backstop for callers using model_construct or a custom request class.
        try:
            if request_image_count(request) > MAX_IMAGE_ATTACHMENTS:
                raise ValueError(f'at most {MAX_IMAGE_ATTACHMENTS} image attachments are supported')
            for image in images or ():
                validate_image_data_url(image)
            for question in request.questions.values():
                for value in question_descriptions(question):
                    for image in description_images(value):
                        validate_image_data_url(image)
        except ValueError as exc:
            raise SchemaError(str(exc)) from None
        if not images:
            return super()._request_state(request)
        return ImageState(_description(request.state), tuple(images))

    def _check_context(self, compiled, state):
        if isinstance(state, ImageState) or compiled.question_images:
            # Tokenizer-only chat templates omit the processor's vision-token
            # expansion. Defer this check to the bounded backend preflight.
            return compiled
        return super()._check_context(compiled, state)

    async def _prepare_jobs(self, jobs, state):
        prepared = {compiled: compiled for compiled, _ in jobs}
        # Sample duplicates still share only request-local evidence. Create a
        # bounded worker pool, not one task per question, while preserving job
        # order and the all-schemas-before-model admission barrier.
        unique = tuple(dict.fromkeys(compiled for compiled, _ in jobs
                                    if isinstance(state, ImageState) or compiled.question_images))
        if not unique:
            return await super()._prepare_jobs(jobs, state)
        pending = iter(unique)

        async def worker():
            for compiled in pending:
                prepared[compiled] = await self._preflight_schema(compiled, state)

        tasks = [asyncio.create_task(worker())
                 for _ in range(min(len(unique), self._image_preflight_concurrency))]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                # Cancelling gather already cancels its children. Do not send a
                # second cancellation into a sibling's in-progress cleanup.
                if not task.done() and not task.cancelling():
                    task.cancel()
            # The caller's read reservation must outlive every preflight and its
            # cleanup, including cancellation or one sibling's invalid evidence.
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return [(prepared[compiled], seed) for compiled, seed in jobs]

    async def _preflight_schema(self, compiled, state):
        body = {
            'model': self.model, 'messages': _messages(compiled, state),
            'add_generation_prompt': True, 'add_special_tokens': False,
            'chat_template_kwargs': {'enable_thinking': False},
        }
        try:
            async with self._image_preflight_slots:
                response = await self._http().post(f'{self.upstream}/tokenize', json=body)
        except httpx.HTTPError as exc:
            raise BackendError('image preprocessing backend could not be reached') from exc
        if response.status_code != 200:
            raise BackendError('image preprocessing backend failed')
        try:
            data = response.json()
            tokens, count, limit = data['tokens'], data['count'], data['max_model_len']
            if (not isinstance(tokens, list) or not tokens
                    or type(count) is not int or count != len(tokens)
                    or type(limit) is not int or limit != self.max_model_len
                    or any(type(token) is not int or not 0 <= token < self._vocab for token in tokens)
                    or IMAGE_TOKEN_ID not in tokens):
                raise ValueError('invalid multimodal preprocessing evidence')
        except (KeyError, TypeError, ValueError) as exc:
            raise BackendError('image preprocessing returned invalid expanded token evidence') from exc
        if count + compiled.canvas_width > self.max_model_len:
            raise SchemaError('the image, state and answer canvas exceed the model context limit; shorten the state or questions')
        return ImageSchema(
            compiled.system_prompt, compiled.template, compiled.slots,
            compiled.canvas_width, compiled.label_ids, count,
            question_images=compiled.question_images,
        )

    async def _read(self, compiled, state, seed):
        result = await super()._read(compiled, state, seed)
        if isinstance(compiled, ImageSchema) and result[2]['prompt_tokens'] != compiled.expanded_prompt_tokens:
            raise BackendError('image inference token accounting differed from expanded preprocessing')
        return result


async def create_engine() -> DjevEngine:
    """Construct the lazy API engine on its serving event loop; no model load."""
    from transformers import AutoTokenizer
    from .config import MODEL, MODEL_REVISION, bounded_integer, get_max_model_len, get_max_request_reads
    tokenizer = await asyncio.to_thread(
        AutoTokenizer.from_pretrained,
        MODEL,
        revision=MODEL_REVISION,
        trust_remote_code=False, local_files_only=os.environ.get('DJEV_OFFLINE', '0') == '1',
    )
    return DjevEngine(
        tokenizer, upstream=os.environ.get('DJEV_UPSTREAM', 'http://127.0.0.1:8001'),
        model='dgemma', canvas=bounded_integer('DJEV_CANVAS', 128, 1, 128),
        compact=os.environ.get('DJEV_COMPACT_CANVAS', '1') == '1',
        max_model_len=get_max_model_len(), max_request_reads=get_max_request_reads(),
        input_transport='token_ids',
    )
