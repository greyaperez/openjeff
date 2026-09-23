import asyncio


import base64


from io import BytesIO


import json


import httpx


from PIL import Image


import pytest


from djev.contracts import DjevRequest


from djev.engine import DiffusionEngine, SchemaError, BackendError


from djev.multimodal import DjevEngine


from test_engine import SmallTokenizer, upstream_response


from test_token_input_transport import RenderedTokenizer, response


async def test_real_factory_pins_model_revision_and_bounds_canvas(monkeypatch):
    import sys
    from types import SimpleNamespace
    from djev.config import MODEL, MODEL_REVISION
    from djev.multimodal import create_engine
    seen = []
    def load(*args, **kwargs):
        seen.append((args, kwargs))
        return RenderedTokenizer()
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=load)))
    monkeypatch.setenv('DJEV_OFFLINE', '1')
    engine = await create_engine()
    assert seen == [((MODEL,), {'revision': MODEL_REVISION, 'trust_remote_code': False, 'local_files_only': True})]
    assert engine.input_transport == 'token_ids' and engine.supports_images
    assert engine.canvas == 128 and engine.compact
    await engine.close()
    monkeypatch.setenv('DJEV_CANVAS', '256')
    with pytest.raises(ValueError, match='DJEV_CANVAS'):
        await create_engine()


def photo(color='red'):
    output = BytesIO()
    Image.new('RGB', (32, 32), color).save(output, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(output.getvalue()).decode()


def request(images=None, **options):
    return DjevRequest.model_validate({'state': {'note': 'inspect the supplied image'},
        'images': [photo()] if images is None else images,
        'questions': {'color': {'type': 'noul', 'instructions': 'Is the image red?'}},
        'options': {'seed': 42, **options}})


def tokenized(count=300, limit=8192):
    return {'tokens': [2, 255999] + [258880]*(count-3) + [106], 'count': count, 'max_model_len': limit}


@pytest.mark.asyncio
async def test_text_only_engine_refuses_to_silently_ignore_images():
    engine = DiffusionEngine(SmallTokenizer())
    with pytest.raises(SchemaError, match='image'):
        await engine.generate(request())


@pytest.mark.asyncio
async def test_image_uses_identical_multimodal_messages_for_real_preflight_and_chat():
    calls=[]
    async def handler(req):
        body=json.loads(req.content);calls.append((req.url.path,body))
        if req.url.path == '/tokenize':
            assert body['chat_template_kwargs'] == {'enable_thinking': False}
            return httpx.Response(200,json=tokenized())
        assert req.url.path == '/v1/chat/completions'
        assert body['messages'] == calls[0][1]['messages']
        assert body['messages'][1]['content'][0] == {'type':'image_url','image_url':{'url':photo()}}
        assert 'prompt' not in body
        result=upstream_response(body['vllm_xargs']['diffusion_seed_canvas'])
        result['usage']['prompt_tokens']=300
        return httpx.Response(200,json=result)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine=DjevEngine(SmallTokenizer(),client=client,input_transport='token_ids')
        result=await engine.generate(request())
        assert result.body['usage']['input_tokens']==300
        assert [p for p,_ in calls]==['/tokenize','/v1/chat/completions']
        assert engine._reserved_reads==0
        assert engine.supports_images is True


@pytest.mark.asyncio
async def test_expanded_count_plus_entire_canvas_rejects_before_inference():
    calls=[]
    async def handler(req):
        calls.append(req.url.path)
        return httpx.Response(200,json=tokenized(8177))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine=DjevEngine(SmallTokenizer(),client=client,compact=True)
        assert engine.compile(request()).canvas_width==16
        with pytest.raises(SchemaError,match='canvas'):
            await engine.generate(request())
        assert calls==['/tokenize'] and engine._reserved_reads==0


@pytest.mark.asyncio
@pytest.mark.parametrize('bad', [
    {'tokens':[2,258880,106],'count':300,'max_model_len':8192},
    {'tokens':[2,True,106],'count':3,'max_model_len':8192},
    {'tokens':[2,77,106],'count':3,'max_model_len':8192},
    {'tokens':[2,258880,106],'count':3,'max_model_len':32768},
])
async def test_invalid_multimodal_preflight_fails_closed(bad):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req:httpx.Response(200,json=bad))) as client:
        with pytest.raises(BackendError):
            await DjevEngine(SmallTokenizer(),client=client).generate(request())


@pytest.mark.asyncio
async def test_no_images_preserves_single_token_completion_request():
    paths=[]
    async def handler(req):
        paths.append(req.url.path);body=json.loads(req.content)
        assert 'messages' not in body
        return httpx.Response(200,json=response(body))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine=DjevEngine(RenderedTokenizer(),client=client,input_transport='token_ids')
        await engine.generate(request(images=[]))
        assert paths==['/v1/completions']


@pytest.mark.asyncio
async def test_image_usage_must_match_actual_expanded_preflight():
    def handler(req):
        body=json.loads(req.content)
        return httpx.Response(200,json=tokenized() if req.url.path=='/tokenize' else upstream_response(body['vllm_xargs']['diffusion_seed_canvas']))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(BackendError,match='token'):
            await DjevEngine(SmallTokenizer(),client=client).generate(request())


@pytest.mark.asyncio
async def test_preflight_cancellation_releases_reservation_without_inference():
    entered=asyncio.Event();paths=[]
    async def handler(req):
        paths.append(req.url.path);entered.set();await asyncio.Event().wait()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine=DjevEngine(SmallTokenizer(),client=client)
        task=asyncio.create_task(engine.generate(request(samples=2)))
        await asyncio.wait_for(entered.wait(),1)
        assert engine._reserved_reads==2
        task.cancel()
        with pytest.raises(asyncio.CancelledError):await task
        assert engine._reserved_reads==0 and paths==['/tokenize']


@pytest.mark.asyncio
async def test_repeated_samples_preflight_once_and_keep_images_out_of_schema_cache():
    paths=[]
    def handler(req):
        paths.append(req.url.path);body=json.loads(req.content)
        if req.url.path=='/tokenize':return httpx.Response(200,json=tokenized())
        data=upstream_response(body['vllm_xargs']['diffusion_seed_canvas']);data['usage']['prompt_tokens']=300
        return httpx.Response(200,json=data)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine=DjevEngine(SmallTokenizer(),client=client)
        await engine.generate(request(samples=3))
        assert paths.count('/tokenize')==1 and paths.count('/v1/chat/completions')==3
        assert all(not hasattr(schema,'expanded_prompt_tokens') for schema in engine._cache.values())
        assert photo() not in repr(engine._cache)


def independent_request():
    payload = request(isolation='independent', diagnostics=True).model_dump(mode='json')
    payload['questions'] = {name: {'type': 'noul', 'instructions': instruction} for name, instruction in (
        ('red', 'Is the image red?'), ('bright', 'Is the scene bright?'), ('square', 'Is the image square?'))}
    return DjevRequest.model_validate(payload)


@pytest.mark.asyncio
async def test_distinct_image_preflights_overlap_but_all_finish_before_any_model_read():
    entered = asyncio.Event(); release = asyncio.Event()
    in_flight = 0; peak = 0; checked = 0; model_reads = 0
    async def handler(req):
        nonlocal in_flight, peak, checked, model_reads
        body = json.loads(req.content)
        if req.url.path == '/tokenize':
            in_flight += 1; peak = max(peak, in_flight)
            if in_flight == 2: entered.set()
            try:
                await release.wait()
                checked += 1
                return httpx.Response(200, json=tokenized())
            finally: in_flight -= 1
        assert checked == 3 and in_flight == 0
        model_reads += 1
        result = upstream_response(body['vllm_xargs']['diffusion_seed_canvas'])
        result['usage']['prompt_tokens'] = 300
        return httpx.Response(200, json=result)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = DjevEngine(SmallTokenizer(), client=client)
        task = asyncio.create_task(engine.generate(independent_request()))
        try:
            await asyncio.wait_for(entered.wait(), .5)
            assert checked == model_reads == 0 and engine._reserved_reads == 3
            release.set()
            await task
        finally:
            release.set()
            if not task.done(): task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        assert peak == 2 and checked == model_reads == 3 and engine._reserved_reads == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['invalid', 'overflow', 'http', 'cancel'])
async def test_concurrent_preflight_failure_awaits_sibling_cleanup_before_releasing_reservation(failure):
    second_entered = asyncio.Event(); fail = asyncio.Event(); cleaning = asyncio.Event(); cleaned = asyncio.Event()
    paths = []; active = 0
    async def handler(req):
        nonlocal active
        paths.append(req.url.path); active += 1
        ordinal = len(paths)
        if ordinal == 2: second_entered.set()
        try:
            if ordinal == 1:
                await fail.wait()
                if failure == 'invalid': return httpx.Response(200, json=tokenized() | {'count': -1})
                if failure == 'overflow': return httpx.Response(200, json=tokenized(8190))
                if failure == 'http': raise httpx.ConnectError('synthetic transport failure')
            await asyncio.Event().wait()
        finally:
            if ordinal == 2:
                cleaning.set()
                await cleaned.wait()
            active -= 1
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = DjevEngine(SmallTokenizer(), client=client)
        task = asyncio.create_task(engine.generate(independent_request()))
        try:
            await asyncio.wait_for(second_entered.wait(), .5)
            if failure == 'cancel': task.cancel()
            else: fail.set()
            await asyncio.wait_for(cleaning.wait(), .5)
            assert not task.done() and engine._reserved_reads == 3
            cleaned.set()
            expected = asyncio.CancelledError if failure == 'cancel' else SchemaError if failure == 'overflow' else BackendError
            with pytest.raises(expected): await task
            assert active == 0 and engine._reserved_reads == 0
            assert paths == ['/tokenize', '/tokenize']
        finally:
            fail.set(); cleaned.set()
            if not task.done(): task.cancel()
            await asyncio.gather(task, return_exceptions=True)
