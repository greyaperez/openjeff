import asyncio


import json


import httpx


import pytest


from pydantic import ValidationError


from djev import contracts


from djev.contracts import DjevRequest


from djev.multimodal import DjevEngine


from djev.engine import BackendError, DiffusionEngine, SchemaError, _messages


from test_multimodal import photo, tokenized


from test_engine import SmallTokenizer, upstream_response


def descriptor(color='red', **extra):
    return {'image': photo(color), **extra}


def payload(questions=None, images=None, **options):
    return {'state': 'Inspect the referenced content.', 'images': images or [],
            'questions': questions or {'q': {'type': 'noul', 'instructions': descriptor()}},
            'options': {'seed': 42, 'isolation': 'independent', **options}}


@pytest.mark.parametrize('question', [
    {'type': 'noul', 'instructions': descriptor(text='🙂' * 2000)},
    {'type': 'noul', 'criteria': {'true': descriptor(text='x' * 500), 'false': 'absent'}},
    {'type': 'choice', 'criteria': {'reference': descriptor(text='x' * 500)}},
    {'type': 'score', 'criteria': [descriptor(text='x' * 500), descriptor('blue')]},
])
def test_all_description_positions_accept_native_images_and_count_only_text(question):
    request = DjevRequest.model_validate(payload({'q': question}))
    assert request.questions['q'].model_dump(exclude_none=True)['type'] == question['type']
    assert contracts.request_has_images(request)
    assert contracts.character_count(descriptor(text='🙂' * 500)) == 500


@pytest.mark.parametrize('attachment', [
    descriptor(extra='forbidden'), descriptor(text=False), descriptor(text=None),
    {'image': 'data:image/png;base64,invalid'}, {'image': 'data:image/svg+xml;base64,PHN2Zy8+'},
    descriptor(text='x' * 2001),
])
def test_malformed_image_descriptions_fail_public_validation(attachment):
    with pytest.raises(ValidationError):
        DjevRequest.model_validate(payload({'q': {'type': 'noul', 'instructions': attachment}}))


def test_attachment_limit_counts_duplicates_and_state_image_separately():
    questions = {'q': {'type': 'choice', 'instructions': descriptor(),
                       'criteria': {str(i): descriptor() for i in range(4)}}}
    request = DjevRequest.model_validate(payload(questions, [photo()]))
    assert contracts.request_image_count(request) == 6
    questions['q']['criteria']['4'] = descriptor()
    with pytest.raises(ValidationError, match='6'):
        DjevRequest.model_validate(payload(questions, [photo()]))
    with pytest.raises(ValidationError):
        DjevRequest.model_validate(payload(images=[photo(), photo()]))


def test_nested_structured_descriptions_are_validated_and_base64_not_counted():
    question = {'type': 'noul', 'instructions': {'reference': [descriptor(text='red')], 'kind': 'compare'}}
    request = DjevRequest.model_validate(payload({'q': question}))
    assert contracts.request_image_count(request) == 1
    assert contracts.character_count(question['instructions']) == len('{"kind":"compare","reference":["red"]}')
    structured = {'image': 'catalog-item-id', 'note': ['existing', 'JSON']}
    assert contracts.character_count(structured) == len(json.dumps(structured, separators=(',', ':'), sort_keys=True))


def image_parts(body):
    return [part['image_url']['url'] for message in body['messages']
            for part in message['content'] if part['type'] == 'image_url']


def mock_answer(body, count=300):
    answer = upstream_response(body['vllm_xargs']['diffusion_seed_canvas'])
    answer['usage']['prompt_tokens'] = count
    answer['usage']['total_tokens'] = count + answer['usage']['completion_tokens']
    return answer


async def test_question_only_image_preflight_matches_inference_and_never_enters_shared_cache():
    calls = []
    async def handler(req):
        body = json.loads(req.content); calls.append((req.url.path, body))
        assert image_parts(body) == [photo()]
        assert 'base64' not in body['messages'][0]['content'][0]['text']
        return httpx.Response(200, json=tokenized() if req.url.path == '/tokenize' else mock_answer(body))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = DjevEngine(SmallTokenizer(), client=client, input_transport='token_ids')
        result = await engine.generate(DjevRequest.model_validate(payload(samples=2)))
        assert [path for path, _ in calls] == ['/tokenize', '/v1/chat/completions', '/v1/chat/completions']
        assert all(body['messages'] == calls[0][1]['messages'] for _, body in calls)
        assert result.body['usage']['input_tokens'] == 600
        assert not engine._cache and engine._reserved_reads == 0


async def test_independent_questions_and_score_children_only_receive_their_own_images():
    state, instruction, first, second, sibling = [photo(c) for c in ['white', 'red', 'blue', 'green', 'yellow']]
    questions = {
        'score': {'type': 'score', 'instructions': {'image': instruction, 'text': 'Reference'},
                  'criteria': [{'image': first, 'text': 'Blue candidate'}, {'image': second, 'text': 'Green candidate'}]},
        'sibling': {'type': 'noul', 'instructions': {'image': sibling}},
    }
    calls = []; preflighted = []
    async def handler(req):
        body = json.loads(req.content); attachments = image_parts(body)
        calls.append((req.url.path, body))
        if req.url.path == '/tokenize':
            preflighted.append(attachments)
            return httpx.Response(200, json=tokenized())
        assert len(preflighted) == 3  # Whole-request barrier, not per-read lazy preflight.
        assert attachments in preflighted
        return httpx.Response(200, json=mock_answer(body))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = DjevEngine(SmallTokenizer(), client=client)
        result = await engine.generate(DjevRequest.model_validate(payload(questions, [state], score_mode='independent_levels')))
    assert preflighted == [[state, instruction, first], [state, instruction, second], [state, sibling]]
    assert result.body['answers']['score']['legend'] == {'0': questions['score']['criteria'][0], '1': questions['score']['criteria'][1]}
    assert result.body['usage']['input_tokens'] == 900


async def test_late_question_image_context_failure_blocks_even_earlier_text_read():
    calls = []
    async def handler(req):
        calls.append(req.url.path)
        return httpx.Response(200, json=tokenized(8177))
    questions = {'text': {'type': 'noul', 'instructions': 'Text only'},
                 'image': {'type': 'noul', 'instructions': descriptor()}}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = DjevEngine(SmallTokenizer(), client=client, compact=True)
        with pytest.raises(SchemaError, match='canvas'):
            await engine.generate(DjevRequest.model_validate(payload(questions)))
        assert calls == ['/tokenize'] and engine._reserved_reads == 0


async def test_question_image_usage_mismatch_is_not_accepted():
    async def handler(req):
        body = json.loads(req.content)
        return httpx.Response(200, json=tokenized() if req.url.path == '/tokenize' else mock_answer(body, 301))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(BackendError, match='accounting'):
            await DjevEngine(SmallTokenizer(), client=client).generate(DjevRequest.model_validate(payload()))


def test_joint_six_images_keep_duplicate_native_parts_and_no_base64_in_prompt():
    question = {'type': 'choice', 'instructions': descriptor(),
                'criteria': {str(i): descriptor() for i in range(4)}}
    request = DjevRequest.model_validate(payload({'q': question}, [photo()], isolation='joint'))
    engine = DjevEngine(SmallTokenizer())
    compiled = engine.compile(request)
    messages = _messages(compiled, engine._request_state(request))
    assert image_parts({'messages': messages}) == [photo()] * 6
    assert 'base64' not in compiled.system_prompt and len(compiled.question_images) == 5
    assert not engine._cache


def test_state_json_image_field_is_not_an_attachment_or_exempt_from_state_limit():
    request = DjevRequest.model_validate(payload({'q': {'type': 'noul'}}) | {'state': descriptor()})
    assert not contracts.request_has_images(request)
    with pytest.raises(ValidationError, match='20000'):
        DjevRequest.model_validate(payload({'q': {'type': 'noul'}}) | {'state': {'image': 'data:' + 'x' * 20000}})
