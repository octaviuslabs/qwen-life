import json

import pytest

from life.internal_runtime import (
    InternalRuntime,
    Protocol,
    consume_records,
    process_queued_command,
    rows_to_cells,
    state_document,
    visible_token_ids,
)


def protocol() -> Protocol:
    return Protocol(
        row_base=262_144,
        reset_token=327_680,
        tick_token=327_681,
        done_token=327_682,
        grid_layer=4,
        base_vocab=248_320,
        model_vocab=327_744,
    )


def test_protocol_checks_aligned_private_vocabulary():
    value = protocol()
    value.validate()
    assert value.is_row(262_144)
    assert value.is_row(327_679)
    assert not value.is_row(value.reset_token)
    assert value.is_private(value.done_token)
    with pytest.raises(ValueError):
        Protocol(
            row_base=262_144,
            reset_token=327_679,
            tick_token=327_681,
            done_token=327_682,
            grid_layer=4,
            base_vocab=248_320,
            model_vocab=327_744,
        ).validate()


def test_protocol_loads_checkpoint_metadata(tmp_path):
    meta = {
        "row_base": 262_144,
        "reset_token": 327_680,
        "tick_token": 327_681,
        "done_token": 327_682,
        "grid_layer": 4,
        "grid_heads": list(range(16)),
        "base_vocab": 248_320,
        "model_vocab": 327_744,
    }
    (tmp_path / "life_internal.json").write_text(json.dumps(meta))
    assert Protocol.load(tmp_path) == protocol()


def test_visible_output_hides_only_transport_and_tool_envelope():
    p = protocol()
    tool_start, tool_end, im_end = 248_058, 248_059, 248_046
    generated = [101, tool_start, 200, 201, tool_end, p.row_base + 7, p.done_token, im_end]
    shown, saw_call = visible_token_ids(generated, p, tool_start, tool_end, im_end)
    assert shown == [101]
    assert saw_call


def test_rows_serialize_without_importing_command_parser():
    rows = (0b101,) + (0,) * 15
    cells = rows_to_cells(rows)
    assert cells[:5] == [1, 0, 1, 0, 0]
    state = state_document(rows, 3, "INTERNAL", "done", 9, [])
    assert state["live_cells"] == 2
    assert state["backend"] == "qwen-life-internal-controller"


def test_non_language_ui_controls_cannot_bypass_qwen():
    runtime = InternalRuntime.__new__(InternalRuntime)
    with pytest.raises(ValueError, match="talk to Qwen"):
        runtime.submit({"type": "clear"})


def test_current_user_message_is_appended_once_to_prompt_history():
    runtime = InternalRuntime.__new__(InternalRuntime)
    runtime.conversation = [
        {"role": "user", "text": "earlier", "kind": "user-chat"},
        {"role": "assistant", "text": "reply", "kind": "chat"},
    ]
    messages = runtime._messages("new request")
    assert [item["content"] for item in messages].count("new request") == 1


def test_hidden_internal_acknowledgements_are_not_controller_few_shots():
    runtime = InternalRuntime.__new__(InternalRuntime)
    runtime.conversation = [
        {"role": "user", "text": "Clear the world", "kind": "user-internal"},
        {
            "role": "assistant",
            "text": "I applied that through my internal neural circuit.",
            "kind": "internal",
        },
        {"role": "user", "text": "How are you?", "kind": "user-chat"},
        {"role": "assistant", "text": "Good.", "kind": "chat"},
    ]
    messages = runtime._messages("Advance one generation")
    contents = [item["content"] for item in messages]
    assert "Clear the world" not in contents
    assert "I applied that through my internal neural circuit." not in contents
    assert "How are you?" in contents
    assert "Good." in contents


def test_controller_system_does_not_carry_old_negation_forward():
    from life.internal_runtime import SYSTEM

    assert "earlier negation must never prevent a later positive command" in SYSTEM


@pytest.mark.parametrize("fail", [False, True])
def test_clear_chat_forgets_prompt_history_and_persists_reset(monkeypatch, tmp_path, fail):
    from life.internal_runtime import TurnResult

    runtime = InternalRuntime.__new__(InternalRuntime)
    runtime.conversation = [
        {"id": 8, "role": "user", "text": "old chat", "kind": "user-chat"},
    ]
    runtime.conversation_id = 8
    runtime.conversation_epoch = 2
    runtime.generation = 5
    runtime.event_id = 10
    runtime.state_path = tmp_path / "state.json"
    monkeypatch.setattr(InternalRuntime, "rows", property(lambda self: (0,) * 16))

    def turn(text):
        assert [item["role"] for item in runtime._messages(text)] == ["system", "user"]
        if fail:
            raise RuntimeError("test operation failed")
        return TurnResult("Cleared", (), (), True, 1, 0)

    monkeypatch.setattr(runtime, "turn", turn)
    error = process_queued_command(runtime, {
        "type": "say", "text": "Clear the entire world now.", "reset_chat": True,
    })
    saved = json.loads(runtime.state_path.read_text())
    assert saved["conversation_epoch"] == 3
    assert saved["generation"] == 5
    assert "old chat" not in json.dumps(saved)
    if fail:
        assert "test operation failed" in error
        assert saved["conversation"][0]["kind"] == "error"
        assert saved["parsed_command"] == "REJECTED"
    else:
        assert error is None
        assert saved["conversation"] == []
        assert runtime.conversation_id == 8


@pytest.mark.parametrize("record_chat", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_control_requests_only_update_status_while_typed_messages_stay_in_chat(
    monkeypatch, tmp_path, record_chat, fail,
):
    from life.internal_runtime import TurnResult

    runtime = InternalRuntime.__new__(InternalRuntime)
    original = [{"id": 1, "role": "user", "text": "My typed message", "kind": "user-chat"}]
    runtime.conversation = list(original)
    runtime.conversation_id = 1
    runtime.conversation_epoch = 0
    runtime.generation = 0
    runtime.event_id = 1
    runtime.state_path = tmp_path / "state.json"
    monkeypatch.setattr(InternalRuntime, "rows", property(lambda self: (0,) * 16))

    def turn(_text):
        if fail:
            raise RuntimeError("operation failed")
        return TurnResult("Advanced", (), (), True, 1, 1)

    monkeypatch.setattr(runtime, "turn", turn)
    command = {"type": "say", "text": "Advance one generation."}
    if not record_chat:
        command["record_chat"] = False
    process_queued_command(runtime, command)
    saved = json.loads(runtime.state_path.read_text())
    assert saved["conversation"][:1] == original
    assert saved["generation"] == (0 if fail else 1)
    assert saved["message"] == ("Command rejected: RuntimeError: operation failed" if fail else "Advanced")
    if record_chat:
        assert len(saved["conversation"]) == (2 if fail else 3)
    else:
        assert saved["conversation"] == original


class _FakeLogits:
    def __init__(self, token):
        self.token = token

    def __getitem__(self, _key):
        return self

    def float(self):
        return self

    def argmax(self):
        return self.token


class _FakeOutput:
    def __init__(self, token, cache):
        self.logits = _FakeLogits(token)
        self.past_key_values = cache


class _FakeModel:
    device = "cpu"

    def __init__(self, tokens):
        self.tokens = iter(tokens)
        self.caches = []

    def __call__(self, **_kwargs):
        cache = object()
        self.caches.append(cache)
        return _FakeOutput(next(self.tokens), cache)


class _FakePrompt:
    def to(self, _device):
        return self


class _FakeTokenizer:
    def apply_chat_template(self, *_args, **_kwargs):
        return {"input_ids": _FakePrompt()}

    def decode(self, *_args, **_kwargs):
        return ""


def _turn_with_trace(monkeypatch, tmp_path, tokens):
    import life.internal_runtime as module

    original = object()
    restored_input = object()
    rolled_back = object()
    restored = []
    saved = []

    def fake_snapshot(cache, _protocol):
        return {"cache": cache}

    def fake_restore(_model, before, _protocol):
        restored.append(before)
        return restored_input if len(restored) == 1 else rolled_back

    monkeypatch.setattr(module, "snapshot_grid", fake_snapshot)
    monkeypatch.setattr(module, "restore_grid", fake_restore)
    monkeypatch.setattr(module, "save_grid", lambda value, _path: saved.append(value))

    runtime = InternalRuntime.__new__(InternalRuntime)
    runtime.protocol = protocol()
    runtime.cache = original
    runtime.model = _FakeModel(tokens)
    runtime.tokenizer = _FakeTokenizer()
    runtime.conversation = []
    runtime.max_new_tokens = 64
    runtime.im_end = 248_046
    runtime.tool_start = 248_058
    runtime.tool_end = 248_059
    runtime.neural_path = tmp_path / "neural.pt"
    result = runtime.turn("test")
    return runtime, result, original, restored_input, rolled_back, restored, saved


def test_turn_rolls_back_grid_when_no_private_rows_commit(monkeypatch, tmp_path):
    runtime, result, original, _input, rolled_back, restored, saved = _turn_with_trace(
        monkeypatch,
        tmp_path,
        [248_058, 248_059, 248_046],
    )
    assert result.tool_call_seen
    assert result.row_commits == 0
    assert runtime.cache is rolled_back
    assert restored == [{"cache": original}, {"cache": original}]
    assert saved == [{"cache": rolled_back}]


def test_turn_keeps_grid_after_complete_private_row_commit(monkeypatch, tmp_path):
    p = protocol()
    trace = [*(p.row_base + value for value in range(16)), p.done_token, 248_046]
    runtime, result, original, _input, _rollback, restored, saved = _turn_with_trace(
        monkeypatch, tmp_path, trace
    )
    assert result.row_commits == 1
    assert runtime.cache is runtime.model.caches[-1]
    assert restored == [{"cache": original}]
    assert saved == [{"cache": runtime.model.caches[-1]}]


def test_rejected_queue_record_is_consumed_and_published():
    class Runtime:
        def __init__(self):
            self.published = None

        def submit(self, command):
            raise ValueError("no direct command")

        def publish(self, parsed, message):
            self.published = parsed, message

    runtime = Runtime()
    error = process_queued_command(runtime, {"type": "clear"})
    assert error.startswith("Command rejected: ValueError:")
    assert runtime.published[0] == "REJECTED"


def test_jsonl_queue_tracks_committed_offsets(tmp_path):
    path = tmp_path / "commands.jsonl"
    path.write_text('{"type":"say","text":"hello"}\n')
    records, offset = consume_records(path, 0)
    assert records[0][0] == {"type": "say", "text": "hello"}
    assert records[0][1] == offset == path.stat().st_size


def test_last_logit_projection_preserves_prefill_and_cached_decode(monkeypatch):
    """The memory-saving projection retains next-token behavior and cache state."""
    import inspect
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    # This test is intentionally CPU-only, including in images with CUDA-only
    # causal-convolution kernels installed. Exercise the reference kernels.
    for name in ("causal_conv1d_fn", "causal_conv1d_update"):
        kernel = getattr(modeling_qwen3_5, name)
        # Newer Transformers wraps a local PyTorch fallback; older versions
        # choose that fallback when the optional CUDA function is absent.
        fallback = inspect.unwrap(kernel) if hasattr(kernel, "__wrapped__") else None
        monkeypatch.setattr(modeling_qwen3_5, name, fallback)

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(19)
            config = Qwen3_5TextConfig(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=16,
                linear_num_key_heads=2,
                linear_num_value_heads=2,
                linear_key_head_dim=8,
                linear_value_head_dim=8,
                layer_types=["linear_attention", "full_attention"],
                max_position_embeddings=64,
                rope_parameters={
                    "rope_type": "default",
                    "rope_theta": 10000,
                    "partial_rotary_factor": 1.0,
                    "mrope_section": [2, 3, 3],
                },
            )
            config._attn_implementation = "eager"
            model = Qwen3_5ForCausalLM(config).eval()
            runtime = InternalRuntime.__new__(InternalRuntime)
            runtime.model = model
            tokens = [1, 5, 9, 2, 7, 3]
            full = model(input_ids=torch.tensor([tokens]), use_cache=True)
            projected_shapes = []
            hook = model.register_forward_hook(
                lambda _model, _args, output: projected_shapes.append(output.logits.shape)
            )
            cache, tail = runtime._feed(tokens)
            hook.remove()
            assert full.logits.shape == (1, len(tokens), config.vocab_size)
            assert projected_shapes == [(1, 1, config.vocab_size)]
            torch.testing.assert_close(tail, full.logits[0, -1], atol=1e-6, rtol=1e-5)

            next_token = int(tail.argmax())
            full_next = model(
                input_ids=torch.tensor([[next_token]]),
                past_key_values=full.past_key_values,
                use_cache=True,
            )
            _cache, next_tail = runtime._feed([next_token], cache)
            torch.testing.assert_close(
                next_tail, full_next.logits[0, -1], atol=1e-6, rtol=1e-5
            )
    finally:
        torch.set_num_threads(previous_threads)
