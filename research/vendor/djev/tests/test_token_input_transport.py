"""Exact token input transport; all upstream responses are CPU test fixtures."""
import asyncio
from copy import deepcopy
import json
import math

import httpx
import pytest

from djev.contracts import DjevRequest
from djev.engine import BackendError, CapacityError, DiffusionEngine, SchemaError
from test_engine import SmallTokenizer


class RenderedTokenizer(SmallTokenizer):
    def __init__(self, length=20, mapping=False):
        self.calls = []
        self.length = length
        self.mapping = mapping

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((deepcopy(messages), kwargs))
        assert kwargs == {
            "tokenize": True, "add_generation_prompt": True, "enable_thinking": False,
        }
        assert all(isinstance(message["content"], list) for message in messages)
        state_id = 222 if messages[1]["content"][0]["text"] == "second state" else 111
        ids = [2, state_id, 0, 106] + [77] * (self.length - 4)
        return {"input_ids": ids} if self.mapping else ids


def payload(*, mode="joint", samples=1, state="first state", all_types=False):
    questions = {"boolean": {"type": "noul", "instructions": "Is it ready?"}}
    if all_types:
        questions.update({
            "route": {"type": "choice", "criteria": {"left": "L", "right": "R"}},
            "rating": {"type": "score", "criteria": ["low", "medium", "high"]},
        })
    return DjevRequest.model_validate({
        "state": state, "questions": questions,
        "options": {
            "seed": -7919, "samples": samples, "diagnostics": True,
            "isolation": "joint" if mode == "joint" else "independent",
            "score_mode": "independent_levels" if mode == "levels" else "categorical",
        },
    })


def response(body, *, completion=True):
    weights = {500: .3, 501: .1, 1065: .12, 1066: .08, 1048: .02, 1049: .04, 1050: .14}
    selected = body["logprob_token_ids"][0]
    scores = {f"token_id:{token}": math.log(weights[token]) for token in body["logprob_token_ids"]}
    rows = body["max_tokens"]
    if completion:
        logprobs = {
            "tokens": [f"token_id:{selected}"] * rows,
            "token_logprobs": [scores[f"token_id:{selected}"]] * rows,
            "top_logprobs": [dict(scores) for _ in range(rows)],
            "text_offset": list(range(rows)),
        }
    else:
        logprobs = {"content": [{
            "token": f"token_id:{selected}", "logprob": scores[f"token_id:{selected}"],
            "top_logprobs": [{"token": key, "logprob": value} for key, value in scores.items()],
        } for _ in range(rows)]}
    return {
        "id": "fixture", "model": "dgemma", "object": "text_completion",
        "choices": [{"index": 0, "text": "ignored", "finish_reason": "length", "logprobs": logprobs}],
        "usage": {"prompt_tokens": 20, "completion_tokens": rows, "total_tokens": 20 + rows},
    }


@pytest.mark.parametrize("transport", [None, True, "completion", "token-id"])
def test_unknown_input_transport_cannot_silently_select_a_backend(transport):
    with pytest.raises(ValueError, match="input_transport"):
        DiffusionEngine(SmallTokenizer(), input_transport=transport)


@pytest.mark.parametrize("mapping", [False, True])
async def test_exact_rendered_ids_are_sent_once_for_multiple_samples(mapping):
    tokenizer = RenderedTokenizer(mapping=mapping)
    sent = []

    async def handler(request):
        body = json.loads(request.content)
        sent.append(body)
        assert request.url.path == "/v1/completions"
        assert body["prompt"] == [2, 111, 0, 106] + [77] * 16
        assert body["logprobs"] == 0 and type(body["logprobs"]) is int
        assert body["echo"] is False and body["add_special_tokens"] is False
        assert not {"messages", "chat_template_kwargs", "top_logprobs"} & body.keys()
        assert body["temperature"] == body["top_p"] == 1 and body["top_k"] == -1
        assert body["return_tokens_as_token_ids"] is True
        assert body["logprob_token_ids"] == [500, 501]
        assert body["vllm_xargs"]["diffusion_max_steps"] == 1
        assert body["vllm_xargs"]["diffusion_read_only"] is True
        return httpx.Response(200, json=response(body))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = DiffusionEngine(tokenizer, client=client, input_transport="token_ids")
        result = await engine.generate(payload(samples=4))
    assert len(tokenizer.calls) == 1 and len(sent) == 4
    assert result.body["answers"]["boolean"]["noul"] == pytest.approx(.75)
    assert result.body["diagnostics"]["questions"]["boolean"]["label_mass"] == pytest.approx(.4)
    assert result.body["usage"] == {"input_tokens": 80, "output_tokens": 4 * sent[0]["max_tokens"]}


@pytest.mark.parametrize("transport,tokenizer", [
    ("chat", RenderedTokenizer), ("token_ids", SmallTokenizer),
])
async def test_default_chat_and_custom_tokenizer_fallback_keep_chat_payload(transport, tokenizer):
    seen = []
    async def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        assert request.url.path == "/v1/chat/completions"
        assert body["logprobs"] is True and body["top_logprobs"] == 0
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert "prompt" not in body and "add_special_tokens" not in body and "echo" not in body
        return httpx.Response(200, json=response(body, completion=False))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        options = {} if transport == "chat" else {"input_transport": transport}
        result = await DiffusionEngine(tokenizer(), client=client, **options).generate(payload())
    assert len(seen) == 1 and result.body["answers"]["boolean"]["noul"] == pytest.approx(.75)


@pytest.mark.parametrize("mode,unique", [("joint", 1), ("independent", 3), ("levels", 5)])
async def test_all_modes_preserve_answers_samples_and_usage_between_transports(mode, unique):
    results, sent_by_transport = [], []
    for transport in ("chat", "token_ids"):
        tokenizer = RenderedTokenizer()
        sent = []
        async def handler(request):
            body = json.loads(request.content)
            sent.append(body)
            return httpx.Response(200, json=response(body, completion=transport == "token_ids"))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            engine = DiffusionEngine(tokenizer, client=client, input_transport=transport)
            results.append((await engine.generate(payload(mode=mode, samples=2, all_types=True))).body)
        assert len(tokenizer.calls) == unique
        assert len(sent) == unique * 2
        sent_by_transport.append(sent)
    assert results[0] == results[1]
    assert [body["vllm_xargs"] for body in sent_by_transport[0]] == [body["vllm_xargs"] for body in sent_by_transport[1]]


@pytest.mark.parametrize("over_by", [0, 1])
async def test_context_reserves_full_canvas_not_only_emitted_tokens(over_by):
    tokenizer = RenderedTokenizer(length=30 + over_by)
    sent = []
    async def handler(request):
        body = json.loads(request.content); sent.append(body)
        return httpx.Response(200, json=response(body))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = DiffusionEngine(tokenizer, client=client, input_transport="token_ids")
        engine.max_model_len = 30 + engine.compile(payload()).canvas_width
        if over_by:
            with pytest.raises(SchemaError, match="context"):
                await engine.generate(payload())
            assert not sent
        else:
            await engine.generate(payload())
            assert len(sent) == 1
            assert sent[0]["max_tokens"] < sent[0]["vllm_xargs"]["diffusion_canvas_length"]


async def test_concurrent_states_do_not_share_prepared_ids_or_cached_answers():
    tokenizer = RenderedTokenizer()
    sent = []
    both_started = asyncio.Event()
    async def handler(request):
        body = json.loads(request.content); sent.append(body)
        if len(sent) >= 2:
            both_started.set()
        await both_started.wait()
        return httpx.Response(200, json=response(body))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = DiffusionEngine(tokenizer, client=client, input_transport="token_ids")
        await asyncio.wait_for(asyncio.gather(
            engine.generate(payload()), engine.generate(payload(state="second state")),
        ), 1)
        await engine.generate(payload())
    assert sorted(body["prompt"][1] for body in sent) == [111, 111, 222]
    assert len(tokenizer.calls) == 3


@pytest.mark.parametrize("ids", [[], [True], [-1], [262144], [1.5], ["2"], [[2, 3]]])
async def test_malformed_rendered_token_ids_fail_before_upstream(ids):
    class MalformedTokenizer(RenderedTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return ids
    sent = []
    async def handler(request):
        sent.append(request)
        return httpx.Response(500)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(SchemaError, match="token IDs"):
            await DiffusionEngine(MalformedTokenizer(), client=client, input_transport="token_ids").generate(payload())
    assert not sent


def corrupt_response(data, failure):
    evidence = data["choices"][0]["logprobs"]
    if failure == "null_evidence":
        data["choices"][0]["logprobs"] = None
    elif failure == "short_rows":
        evidence["tokens"] = []
        evidence["token_logprobs"] = []
        evidence["top_logprobs"] = []
    elif failure == "misaligned":
        evidence["tokens"].pop()
    elif failure == "not_arrays":
        evidence["top_logprobs"] = {}
    elif failure == "null_row":
        evidence["top_logprobs"] = [None] * len(evidence["tokens"])
    elif failure == "row_list":
        evidence["top_logprobs"] = [[]] * len(evidence["tokens"])
    elif failure == "missing_label":
        for row in evidence["top_logprobs"]:
            del row["token_id:501"]
    elif failure == "string_label":
        for row in evidence["top_logprobs"]:
            row["no"] = row.pop("token_id:501")
    elif failure == "bad_token_id":
        for row in evidence["top_logprobs"]:
            row["token_id:bad"] = row.pop("token_id:501")
    elif failure in {"nan", "infinite", "positive", "not_number"}:
        value = {"nan": "NaN", "infinite": "Infinity", "positive": .1, "not_number": None}[failure]
        for row in evidence["top_logprobs"]:
            row["token_id:501"] = value
    elif failure == "selected_invalid":
        evidence["token_logprobs"] = [None] * len(evidence["tokens"])
    elif failure == "selected_disagrees":
        evidence["token_logprobs"] = [-4.0] * len(evidence["tokens"])
    elif failure == "selected_not_id":
        evidence["tokens"] = [500] * len(evidence["tokens"])
    elif failure in {"negative_id", "out_of_vocab_id", "noncanonical_id"}:
        token = {"negative_id": "token_id:-1", "out_of_vocab_id": "token_id:262144",
                 "noncanonical_id": "token_id:0500"}[failure]
        evidence["tokens"] = [token] * len(evidence["tokens"])
    elif failure.startswith("usage_"):
        data["usage"]["prompt_tokens"] = {"usage_bool": True, "usage_string": "20", "usage_negative": -1}[failure]
    elif failure == "not_object":
        return []
    else:
        raise AssertionError(failure)
    return data


@pytest.mark.parametrize("failure", [
    "null_evidence", "short_rows", "misaligned", "not_arrays", "null_row", "row_list",
    "missing_label", "string_label", "bad_token_id", "nan", "infinite", "positive",
    "not_number", "selected_invalid", "selected_disagrees", "selected_not_id",
    "negative_id", "out_of_vocab_id", "noncanonical_id",
    "usage_bool", "usage_string", "usage_negative", "not_object",
])
async def test_malformed_completion_evidence_is_rejected_and_releases_budget(failure):
    async def handler(request):
        body = json.loads(request.content)
        return httpx.Response(200, json=corrupt_response(response(body), failure))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = DiffusionEngine(RenderedTokenizer(), client=client, input_transport="token_ids")
        with pytest.raises(BackendError):
            await engine.generate(payload())
        assert engine._reserved_reads == 0


@pytest.mark.parametrize("all_censored", [False, True])
async def test_completion_censoring_uses_impossible_evidence_not_a_fake_finite_logprob(all_censored):
    async def handler(request):
        body = json.loads(request.content)
        data = response(body)
        evidence = data["choices"][0]["logprobs"]
        for row in evidence["top_logprobs"]:
            row["token_id:501"] = -9999.0
            if all_censored:
                row["token_id:500"] = -9999.0
        if all_censored:
            evidence["token_logprobs"] = [-9999.0] * len(evidence["tokens"])
        return httpx.Response(200, json=data)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = DiffusionEngine(RenderedTokenizer(), client=client, input_transport="token_ids")
        if all_censored:
            with pytest.raises(BackendError):
                await engine.generate(payload())
        else:
            result = await engine.generate(payload())
            assert result.body["answers"]["boolean"]["noul"] == 1.0
            assert result.body["diagnostics"]["questions"]["boolean"]["label_mass"] == pytest.approx(.3)


async def test_late_question_context_failure_prevents_all_completion_reads():
    class LengthTokenizer(RenderedTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            if "high" in messages[0]["content"][0]["text"]:
                return [77] * 8192
            return super().apply_chat_template(messages, **kwargs)
    sent = []
    async def handler(request):
        sent.append(request)
        return httpx.Response(500)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = DiffusionEngine(LengthTokenizer(), client=client, input_transport="token_ids")
        with pytest.raises(SchemaError, match="context"):
            await engine.generate(payload(mode="independent", all_types=True))
    assert not sent


@pytest.mark.parametrize("mode", ["joint", "independent", "levels"])
async def test_token_transport_keeps_bounded_admission_and_cancels_all_siblings(mode):
    entered = asyncio.Event()
    held = asyncio.Event()
    active = 0
    peak = 0
    async def handler(request):
        nonlocal active, peak
        assert request.url.path == "/v1/completions"
        active += 1; peak = max(peak, active)
        if active == 2:
            entered.set()
        try:
            await held.wait()
        finally:
            active -= 1
        raise AssertionError("cancelled request should not return")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = DiffusionEngine(RenderedTokenizer(), client=client, input_transport="token_ids",
                                 max_active_reads=2, max_request_reads=2, max_reserved_reads=2)
        running = asyncio.create_task(engine.generate(payload(mode=mode, samples=2)))
        await asyncio.wait_for(entered.wait(), 1)
        try:
            with pytest.raises(CapacityError):
                await engine.generate(payload(mode=mode))
        finally:
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
        assert peak == 2 and active == 0 and engine._reserved_reads == 0


@pytest.mark.parametrize("failure", ["status", "connection", "invalid_json"])
async def test_completion_transport_errors_do_not_reflect_private_payload(failure):
    async def handler(request):
        if failure == "connection":
            raise httpx.ConnectError("private prompt or credentials", request=request)
        return httpx.Response(503 if failure == "status" else 200, text="private prompt or credentials")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = DiffusionEngine(RenderedTokenizer(), client=client, input_transport="token_ids")
        with pytest.raises(BackendError) as caught:
            await engine.generate(payload())
        assert "private" not in str(caught.value) and engine._reserved_reads == 0
