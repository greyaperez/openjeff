from copy import deepcopy
import json
from types import SimpleNamespace as NS

import pytest

from runtime import install, serve, vision


def config():
    return NS(hf_config=NS(model_type="diffusion_gemma"),
              hf_text_config=NS(use_bidirectional_attention="vision", sliding_window=1024),
              max_model_len=32768)


def image(begin=3, end=5):
    return NS(modality="image", mm_position=NS(extract_embeds_range=lambda: [(begin, end)]))


def params():
    return NS(extra_args={"diffusion_canvas_length": 16, "diffusion_read_only": True,
                          "diffusion_max_steps": 1}, skip_reading_prefix_cache=False)


def test_image_admission_preserves_parameters_but_bypasses_text_kv_reuse():
    p = params()
    original = deepcopy(p)
    vision.admit(config(), NS(max_num_batched_tokens=2048), [image()], p, 100, 128)
    assert p.skip_reading_prefix_cache and p.extra_args == original.extra_args
    p = params()
    vision.admit(config(), NS(max_num_batched_tokens=2048), [], p, 100, 128)
    assert p == original


def test_multiple_images_sort_coordinates_and_reject_overlapping_embeddings():
    assert vision.image_spans(config(), [image(20, 30), image(3, 5)], 40) == [(3, 5), (20, 30)]
    with pytest.raises(ValueError, match="overlap"):
        vision.image_spans(config(), [image(3, 5), image(5, 8)], 40)
    with pytest.raises(ValueError, match="six"):
        vision.image_spans(config(), [image(i * 10, i * 10 + 5) for i in range(7)], 100)


@pytest.mark.parametrize("start,count,wanted", [(0, 4, 3), (3, 3, 3), (0, 6, 6), (6, 2, 2)])
def test_scheduling_never_splits_inside_image_embeddings(start, count, wanted):
    request = NS(mm_features=[image()], num_prompt_tokens=20)
    assert vision.atomic_image_budget(config(), request, start, count) == wanted


def test_partial_image_reentry_and_insufficient_budget_fail_before_dispatch():
    with pytest.raises(ValueError, match="restart"):
        vision.atomic_image_budget(config(), NS(mm_features=[image()], num_prompt_tokens=20), 4, 1)
    with pytest.raises(ValueError, match="budget"):
        vision.admit(config(), NS(max_num_batched_tokens=2), [image()], params(), 20, 128)


def test_image_overlay_uses_actual_mixed_batch_row_and_refuses_capture():
    state = NS(model_config=config(), encoder_cache=NS(mm_features={"visual": [image()]}))
    batch = NS(req_ids=["text", "visual", "decode"], is_prefilling_np=[True, True, False],
               prefill_len_np=[20, 20, 20], num_computed_prefill_tokens_np=[0, 0, 20],
               num_scheduled_tokens=[6, 6, 1])
    assert vision.ranges_for_batch(state, batch) == {1: [(3, 5)]}
    assert vision.needs_eager(state, batch)
    with pytest.raises(ValueError, match="capture"):
        vision.ranges_for_batch(state, batch, for_capture=True)


def test_distributed_image_configuration_is_rejected():
    parallel = NS(tensor_parallel_size=2, pipeline_parallel_size=1, data_parallel_size=1,
                  decode_context_parallel_size=1, prefill_context_parallel_size=1, use_ubatching=False)
    cfg = NS(model_config=config(), parallel_config=parallel, num_prefill_lookahead_tokens=0,
             speculative_config=None, kv_transfer_config=None, ec_transfer_config=None)
    with pytest.raises(ValueError, match="one worker"):
        vision.validate_configuration(cfg, [image()])


def test_runtime_command_keeps_bf16_and_backend_loopback():
    command = serve.command()
    for option, value in (("--dtype", "bfloat16"), ("--kv-cache-dtype", "bfloat16"),
                          ("--host", "127.0.0.1"), ("--port", "8001"),
                          ("--max-num-batched-tokens", "2048")):
        assert command[command.index(option) + 1] == value
    assert "--quantization" not in command and "--mm-encoder-attn-dtype" not in command


def test_source_mismatch_is_rejected_before_any_download():
    def forbidden(url):
        raise AssertionError("Do not download after a base mismatch")
    with pytest.raises(ValueError, match="base source hash"):
        install.build_sources({name: b"unexpected" for name in install.base_hashes()}, fetch=forbidden)


def test_installer_does_not_write_until_entire_candidate_is_valid(tmp_path, monkeypatch):
    path = tmp_path / "source.py"
    path.write_bytes(b"original")
    monkeypatch.setattr(install.importlib.metadata, "version", lambda name: install.MANIFEST["packages"][name])
    monkeypatch.setattr(install, "base_hashes", lambda: {"source.py": install.sha(b"original")})
    def fail(*args, **kwargs):
        raise ValueError("candidate rejected")
    monkeypatch.setattr(install, "build_sources", fail)
    with pytest.raises(ValueError, match="candidate rejected"):
        install.install(tmp_path)
    assert path.read_bytes() == b"original"
    assert not (tmp_path / "djev_sources.json").exists()


def test_installer_records_exact_candidates_and_removes_old_bytecode(tmp_path, monkeypatch):
    source = tmp_path / "source.py"
    source.write_bytes(b"original")
    source.with_suffix(".pyc").write_bytes(b"stale")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    cached = cache / "source.cpython-312.pyc"
    cached.write_bytes(b"stale")
    monkeypatch.setattr(install.importlib.metadata, "version", lambda name: install.MANIFEST["packages"][name])
    monkeypatch.setattr(install, "base_hashes", lambda: {"source.py": install.sha(b"original")})
    candidate = install.MODIFIED_SOURCE_NOTICE + b"# SPDX-License-Identifier: Apache-2.0\nvalue = 1\n"
    def build(originals, **kwargs):
        assert originals == {"source.py": b"original"}
        return {"source.py": candidate}
    monkeypatch.setattr(install, "build_sources", build)
    proof = install.install(tmp_path)
    assert source.read_bytes() == candidate
    assert b"SPDX-License-Identifier: Apache-2.0" in source.read_bytes()
    assert not source.with_suffix(".pyc").exists() and not cached.exists()
    assert proof["files"] == {"source.py": install.sha(candidate)}
    assert json.loads((tmp_path / "djev_sources.json").read_text()) == proof
