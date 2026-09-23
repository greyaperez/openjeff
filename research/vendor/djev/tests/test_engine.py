import json


import math


import re


import asyncio


import httpx


import pytest


from djev.contracts import DjevRequest


from djev.engine import BackendError, DiffusionEngine, SchemaError


class SmallTokenizer:
    """An explicit tokenizer double; GPU/model access is not part of unit tests."""
    vocab_size = 262144

    def encode(self, text, add_special_tokens=False):
        mapping = {"yes": 500, "no": 501, "<|channel>": 100, "<channel|>": 101}
        parts = re.findall(r"<\|channel>|<channel\|>|yes|no|.", text, re.DOTALL)
        return [mapping.get(p, ord(p) + 1000 if len(p) == 1 else 999) for p in parts]


def request(**options):
    return DjevRequest.model_validate({
        "state": {"text": "Cancel my plan"},
        "questions": {"hidden customer name": {"type": "noul", "instructions": "Wants cancellation?"}},
        "options": options,
    })


def upstream_response(canvas, missing=False):
    rows = []
    for _ in canvas:
        scores = [{"token": "token_id:501", "logprob": math.log(.1)}]
        if not missing:
            scores.append({"token": "token_id:500", "logprob": math.log(.3)})
        rows.append({"token": "token_id:999", "logprob": math.log(.5), "top_logprobs": scores})
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "logprobs": {"content": rows}, "finish_reason": "length"}], "usage": {"prompt_tokens": 20, "completion_tokens": len(canvas), "total_tokens": 20 + len(canvas)}}


@pytest.mark.asyncio
async def test_exact_logprobs_produce_noul_and_mass_without_leaking_ids():
    async def handler(http_request):
        body = json.loads(http_request.content)
        assert body["model"] == "dgemma"
        assert "hidden customer name" not in body["messages"][0]["content"][0]["text"]
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert body["vllm_xargs"]["diffusion_read_only"] is True
        assert body["vllm_xargs"]["diffusion_max_steps"] == 1
        assert body["top_p"] == 1 and body["top_k"] == -1
        assert set(body["logprob_token_ids"]) == {500, 501}
        return httpx.Response(200, json=upstream_response(body["vllm_xargs"]["diffusion_seed_canvas"]))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    engine = DiffusionEngine(tokenizer=SmallTokenizer(), client=client)
    result = await engine.generate(request(diagnostics=True, seed=7))
    assert result.body["answers"]["hidden customer name"]["noul"] == pytest.approx(.75)
    assert result.body["diagnostics"]["questions"]["hidden customer name"]["label_mass"] == pytest.approx(.4)
    assert result.body["diagnostics"]["calibration"] == "unvalidated"
    assert result.body["usage"]["input_tokens"] == 20
    assert result.body["model"] == "djev-0.1"
    assert set(result.body["usage"]) == {"input_tokens", "output_tokens"}
    assert result.compile_ms >= 0 and result.model_ms > 0
    await client.aclose()


@pytest.mark.asyncio
async def test_missing_exact_label_fails_instead_of_fabricating_confidence():
    async def handler(http_request):
        body = json.loads(http_request.content)
        return httpx.Response(200, json=upstream_response(body["vllm_xargs"]["diffusion_seed_canvas"], missing=True))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = DiffusionEngine(tokenizer=SmallTokenizer(), client=client)
        with pytest.raises(BackendError, match="label"):
            await engine.generate(request())


def test_template_must_fit_before_any_gpu_work():
    engine = DiffusionEngine(tokenizer=SmallTokenizer(), canvas=8)
    with pytest.raises(SchemaError, match="canvas"):
        engine.compile(request())


def test_context_sensitive_token_length_is_rejected():
    class BadTokenizer(SmallTokenizer):
        def encode(self, text, add_special_tokens=False):
            result = super().encode(text, add_special_tokens=add_special_tokens)
            return result + ([700] if "yes" in text else [])

    engine = DiffusionEngine(tokenizer=BadTokenizer())
    with pytest.raises(SchemaError, match="single token"):
        engine.compile(request())


@pytest.mark.asyncio
async def test_fixed_seed_repeats_canvas_but_never_caches_model_answer():
    canvases = []
    async def handler(http_request):
        body = json.loads(http_request.content)
        canvas = body["vllm_xargs"]["diffusion_seed_canvas"]
        canvases.append(canvas)
        return httpx.Response(200, json=upstream_response(canvas))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = DiffusionEngine(tokenizer=SmallTokenizer(), client=client)
        await engine.generate(request(seed=17))
        await engine.generate(request(seed=17))
    assert len(canvases) == 2
    assert canvases[0] == canvases[1]
    assert len(canvases[0]) % 16 == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("isolation", ["joint", "independent"])
async def test_multiple_samples_average_distributions_and_sum_actual_usage(isolation):
    n = 0
    async def handler(http_request):
        nonlocal n
        n += 1
        body = json.loads(http_request.content)
        response = upstream_response(body["vllm_xargs"]["diffusion_seed_canvas"])
        if n == 2:
            for row in response["choices"][0]["logprobs"]["content"]:
                row["top_logprobs"][0]["logprob"] = math.log(.3)
                row["top_logprobs"][1]["logprob"] = math.log(.1)
        return httpx.Response(200, json=response)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = DiffusionEngine(tokenizer=SmallTokenizer(), client=client)
        result = await engine.generate(request(samples=2, isolation=isolation))
    assert result.body["answers"]["hidden customer name"]["noul"] == pytest.approx(.5)
    assert result.body["usage"]["input_tokens"] == 40


@pytest.mark.asyncio
async def test_upstream_error_is_explicit_and_does_not_leak_response_body():
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(500, text="private model details"))) as client:
        engine = DiffusionEngine(tokenizer=SmallTokenizer(), client=client)
        with pytest.raises(BackendError) as exc:
            await engine.generate(request())
        assert "private model details" not in str(exc.value)
        assert not await engine.health()


@pytest.mark.asyncio
async def test_choice_order_and_score_levels_remain_aligned():
    payload = DjevRequest.model_validate({
        "state": "example",
        "questions": {
            "route": {"type": "choice", "criteria": {"billing": "Payments", "support": "Help"}},
            "severity": {"type": "score", "criteria": ["low", "medium", "high"]},
        },
    })
    async def handler(http_request):
        body = json.loads(http_request.content)
        canvas = body["vllm_xargs"]["diffusion_seed_canvas"]
        assert set(body["logprob_token_ids"]) == {1065, 1066, 1048, 1049, 1050}
        labels = [(1065, .2), (1066, .8), (1048, .1), (1049, .2), (1050, .7)]
        # Each model row has its own distribution; provide complete exact labels.
        response = upstream_response(canvas)
        for row in response["choices"][0]["logprobs"]["content"]:
            row["top_logprobs"] = [{"token": f"token_id:{i}", "logprob": math.log(p)} for i, p in labels]
        return httpx.Response(200, json=response)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await DiffusionEngine(SmallTokenizer(), client=client).generate(payload)
    assert result.body["answers"]["route"]["choice"] == "support"
    assert result.body["answers"]["route"]["probabilities"] == pytest.approx({"billing": .2, "support": .8})
    assert result.body["answers"]["severity"]["score"] == pytest.approx(1.6)


def test_single_choice_still_has_a_scored_token_slot():
    payload = DjevRequest.model_validate({"state": "example", "questions": {"route": {"type": "choice", "criteria": {"support": None}}}})
    compiled = DiffusionEngine(SmallTokenizer()).compile(payload)
    assert compiled.slots[0].token_ids == (1065,)


@pytest.mark.asyncio
@pytest.mark.parametrize("isolation", ["joint", "independent"])
async def test_failed_sample_cancels_and_awaits_other_inflight_reads(isolation):
    import asyncio
    second_started = asyncio.Event()
    second_stopped = asyncio.Event()
    count = 0
    async def handler(http_request):
        nonlocal count
        count += 1
        if count == 1:
            await second_started.wait()
            return httpx.Response(500)
        second_started.set()
        try:
            await asyncio.sleep(10)
        finally:
            second_stopped.set()
        raise AssertionError("the sibling read should have been cancelled")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = DiffusionEngine(SmallTokenizer(), client=client)
        with pytest.raises(BackendError):
            await engine.generate(request(samples=2, isolation=isolation))
        assert second_stopped.is_set()
        assert engine._reserved_reads == 0


@pytest.mark.asyncio
async def test_all_censored_label_scores_fail_instead_of_uniform_fallback():
    async def handler(http_request):
        body = json.loads(http_request.content)
        response = upstream_response(body["vllm_xargs"]["diffusion_seed_canvas"])
        for row in response["choices"][0]["logprobs"]["content"]:
            for score in row["top_logprobs"]:
                score["logprob"] = -9999.0
        return httpx.Response(200, json=response)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = DiffusionEngine(SmallTokenizer(), client=client)
        with pytest.raises(BackendError):
            await engine.generate(request())


@pytest.mark.asyncio
async def test_negative_seed_samples_do_not_mirror_into_duplicate_canvases():
    canvases = []
    async def handler(http_request):
        body = json.loads(http_request.content)
        canvas = body["vllm_xargs"]["diffusion_seed_canvas"]
        canvases.append(tuple(canvas))
        return httpx.Response(200, json=upstream_response(canvas))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await DiffusionEngine(SmallTokenizer(), client=client).generate(request(seed=-7919, samples=3))
    assert len(set(canvases)) == 3


async def test_independent_focal_input_ignores_ids_order_siblings_and_sample_count():
    focal = {"type": "noul", "instructions": "FOCAL_MARKER"}
    other = {"type": "choice", "instructions": "OTHER_MARKER", "criteria": {"a": None, "b": None}}
    recorded = []
    async def handler(http_request):
        body = json.loads(http_request.content)
        if "FOCAL_MARKER" in body["messages"][0]["content"][0]["text"]:
            assert "OTHER_MARKER" not in body["messages"][0]["content"][0]["text"]
            assert "Question 1:" not in body["messages"][0]["content"][0]["text"]
            recorded.append(body)
        return httpx.Response(200, json=exact_response(body))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = DiffusionEngine(SmallTokenizer(), client=client)
        for questions in ({"original": focal}, {"other": other, "renamed": focal},
                          {"renamed_again": focal, "other": other}, {"a": focal, "duplicate": focal}):
            await engine.generate(independent_request(questions))
        assert len(recorded) == 4
        assert recorded == [recorded[0]] * 4
        recorded.clear()
        await engine.generate(independent_request({"q": focal}, samples=1, seed=-(10**40)))
        first = recorded.pop()
        await engine.generate(independent_request({"renamed": focal}, samples=4, seed=-(10**40)))
        assert len(recorded) == 4 and recorded[0] == first


async def test_independent_preflights_every_question_before_sending_any_read():
    class ContextTokenizer(SmallTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return list(range(8190 if "too_long" in messages[0]["content"][0]["text"] else 20))
    calls = 0
    async def handler(http_request):
        nonlocal calls
        calls += 1
        return httpx.Response(500)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = DiffusionEngine(ContextTokenizer(), client=client)
        with pytest.raises(SchemaError, match="context"):
            await engine.generate(independent_request({"first": {"type": "noul"}, "last": {"type": "noul", "instructions": "too_long"}}))
    assert calls == 0


@pytest.mark.parametrize("request_workers", [4, 8])
async def test_independent_bounds_global_reads_and_per_request_fanout(request_workers):
    active = 0; peak = 0; per_state = {}; per_state_peak = {}
    full = asyncio.Event(); release = asyncio.Event()
    async def handler(http_request):
        nonlocal active, peak
        body = json.loads(http_request.content);state = body["messages"][1]["content"][0]["text"]
        active += 1;peak = max(peak, active)
        per_state[state] = per_state.get(state, 0) + 1
        per_state_peak[state] = max(per_state_peak.get(state, 0), per_state[state])
        if active == 8:full.set()
        try:
            await release.wait()
            return httpx.Response(200, json=exact_response(body))
        finally:
            active -= 1;per_state[state] -= 1
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = DiffusionEngine(SmallTokenizer(), client=client, max_request_reads=request_workers)
        payload = independent_request({str(i): {"type": "noul", "instructions": f"question {i}"} for i in range(8)})
        tasks = [asyncio.create_task(engine.generate(payload.model_copy(update={"state": str(i)}))) for i in range(3)]
        try:
            async with asyncio.timeout(1):await full.wait()
            assert peak == 8 and max(per_state_peak.values()) == request_workers
        finally:
            release.set()
            await asyncio.gather(*tasks)
        assert peak == 8 and max(per_state_peak.values()) == request_workers and active == 0


async def test_read_reservation_overload_is_atomic_and_released_after_failure():
    from djev.engine import CapacityError
    entered = asyncio.Event();release = asyncio.Event();calls = 0
    async def handler(http_request):
        nonlocal calls
        calls += 1;entered.set();await release.wait()
        return httpx.Response(500)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = DiffusionEngine(SmallTokenizer(), client=client, max_reserved_reads=2)
        first = asyncio.create_task(engine.generate(request(samples=2)))
        await entered.wait()
        try:
            with pytest.raises(CapacityError):
                await engine.generate(independent_request({"q": {"type": "noul"}}))
            assert calls == 2
        finally:
            release.set()
            with pytest.raises(BackendError):await first
        assert engine._reserved_reads == 0
        with pytest.raises(BackendError):await engine.generate(request())
        assert calls == 3



def independent_request(questions, **options):
    return DjevRequest.model_validate({
        "state": "shared state", "questions": questions,
        "options": {"isolation": "independent", "seed": -7919, **options},
    })


def exact_response(body, weights=None):
    ids = body["logprob_token_ids"]
    weights = weights or {token: 1 / len(ids) for token in ids}
    entries = [{"token": f"token_id:{token}", "logprob": math.log(weights[token])} for token in ids]
    rows = [{"token": entries[0]["token"], "logprob": entries[0]["logprob"], "top_logprobs": entries}
            for _ in body["vllm_xargs"]["diffusion_seed_canvas"]]
    return {"choices": [{"logprobs": {"content": rows}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": len(rows), "total_tokens": 20 + len(rows)}}
