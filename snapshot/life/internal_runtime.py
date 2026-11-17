"""Generic inference loop for the directly programmed Life checkpoint.

This runtime does not interpret natural language, choose an operation, parse a
Life command, or compute a board update.  Qwen emits its native tool-call token
stream and analytically constructed layers inside the checkpoint consume it.
Those layers emit private row/control tokens while they update recurrent neural
state.  The runtime's responsibilities are deliberately generic: autoregressive
decoding, hiding private transport tokens, snapshotting recurrent state, and
publishing a renderable grid.

The companion checkpoint must contain ``life_internal.json``.  In particular,
``grid_layer`` identifies the DeltaNet layer whose first sixteen value heads
hold the sixteen packed rows.  This keeps the runtime independent of the number
of parser/executor layers selected by the weight compiler.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoTokenizer, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from .internal_tools import CONTROLLER_SYSTEM, TOOLS
from .reference import INITIAL_ROWS


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME = PROJECT / "ui" / "internal-runtime"
META_NAME = "life_internal.json"

SYSTEM = CONTROLLER_SYSTEM + """
Never emit two function calls in one turn. The function executor is part of your own neural
network, so invoke it exactly as you would an internal capability."""


@dataclass(frozen=True)
class Protocol:
    row_base: int
    reset_token: int
    tick_token: int
    done_token: int
    grid_layer: int
    base_vocab: int
    model_vocab: int
    grid_heads: tuple[int, ...] = tuple(range(16))

    @classmethod
    def load(cls, model_path: Path) -> "Protocol":
        path = Path(model_path) / META_NAME
        value = json.loads(path.read_text(encoding="utf-8"))
        required = {
            "row_base",
            "reset_token",
            "tick_token",
            "done_token",
            "grid_layer",
            "base_vocab",
            "model_vocab",
        }
        missing = required - value.keys()
        if missing:
            raise ValueError(f"{META_NAME} is missing {sorted(missing)}")
        protocol = cls(
            row_base=int(value["row_base"]),
            reset_token=int(value["reset_token"]),
            tick_token=int(value["tick_token"]),
            done_token=int(value["done_token"]),
            grid_layer=int(value["grid_layer"]),
            base_vocab=int(value["base_vocab"]),
            model_vocab=int(value["model_vocab"]),
            grid_heads=tuple(int(x) for x in value.get("grid_heads", range(16))),
        )
        protocol.validate()
        return protocol

    def validate(self) -> None:
        if len(self.grid_heads) != 16 or len(set(self.grid_heads)) != 16:
            raise ValueError("the neural grid must use sixteen distinct heads")
        if not (0 <= self.base_vocab <= self.row_base < self.model_vocab):
            raise ValueError("invalid base/row/model vocabulary boundaries")
        if self.reset_token != self.row_base + (1 << 16):
            raise ValueError("RESET must immediately follow the 65536 packed row tokens")
        if (self.tick_token, self.done_token) != (
            self.reset_token + 1,
            self.reset_token + 2,
        ):
            raise ValueError("TICK and DONE must immediately follow RESET")
        if self.done_token >= self.model_vocab or self.grid_layer < 0:
            raise ValueError("protocol token/layer is outside the checkpoint")

    def is_row(self, token: int) -> bool:
        return self.row_base <= int(token) < self.reset_token

    def is_private(self, token: int) -> bool:
        return self.row_base <= int(token) <= self.done_token


@dataclass(frozen=True)
class TurnResult:
    visible: str
    generated: tuple[int, ...]
    private: tuple[int, ...]
    tool_call_seen: bool
    row_commits: int
    tick_tokens: int

    @property
    def changed_through_circuit(self) -> bool:
        return self.row_commits > 0


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def rows_to_cells(rows: Iterable[int]) -> list[int]:
    rows = tuple(int(row) for row in rows)
    if len(rows) != 16 or any(not 0 <= row < 1 << 16 for row in rows):
        raise ValueError("expected sixteen unsigned 16-bit neural rows")
    return [(rows[y] >> x) & 1 for y in range(16) for x in range(16)]


def state_document(
    rows: tuple[int, ...],
    generation: int,
    parsed: str,
    message: str,
    event_id: int,
    conversation: list[dict[str, Any]],
    conversation_epoch: int = 0,
) -> dict[str, Any]:
    cells = rows_to_cells(rows)
    return {
        "width": 16,
        "height": 16,
        "cells": cells,
        "generation": int(generation),
        "live_cells": sum(cells),
        "backend": "qwen-life-internal-controller",
        "parsed_command": parsed,
        "message": message,
        "event_id": int(event_id),
        "conversation": list(conversation),
        "conversation_epoch": int(conversation_epoch),
    }


def _grid_state(cache, protocol: Protocol):
    layer = cache.layers[protocol.grid_layer]
    states = getattr(layer, "recurrent_states", None)
    if not states or states[0] is None:
        return None
    return states[0][0]


def read_rows(cache, protocol: Protocol) -> tuple[int, ...]:
    """Read the packed board directly from recurrent DeltaNet state."""
    state = _grid_state(cache, protocol)
    if state is None:
        return (0,) * 16
    rows = []
    for head in protocol.grid_heads:
        row = state[head, 0]
        rows.append(sum(int(row[2 + x] > 0) << x for x in range(16)))
    return tuple(rows)


def snapshot_grid(cache, protocol: Protocol) -> dict[str, Any]:
    state = _grid_state(cache, protocol)
    if state is None:
        raise ValueError("cache has no initialized internal Life state")
    return {
        "layer": protocol.grid_layer,
        "heads": list(protocol.grid_heads),
        "states": state[list(protocol.grid_heads)].detach().clone().cpu(),
    }


def restore_grid(model, saved: dict[str, Any], protocol: Protocol, batch: int = 1):
    """Restore only authoritative world heads into an otherwise fresh cache."""
    if int(saved["layer"]) != protocol.grid_layer:
        raise ValueError("snapshot grid layer does not match this checkpoint")
    if tuple(int(x) for x in saved["heads"]) != protocol.grid_heads:
        raise ValueError("snapshot grid heads do not match this checkpoint")
    cfg = model.config.get_text_config(decoder=True)
    cache = DynamicCache(config=model.config)
    device = next(model.parameters()).device
    layer = cache.layers[protocol.grid_layer]
    recurrent_dtype = torch.promote_types(saved["states"].dtype, torch.float32)
    recurrent = torch.zeros(
        batch,
        cfg.linear_num_value_heads,
        cfg.linear_key_head_dim,
        cfg.linear_value_head_dim,
        dtype=recurrent_dtype,
        device=device,
    )
    for head, state in zip(protocol.grid_heads, saved["states"]):
        recurrent[:, head] = state.to(device)
    conv_dim = (
        2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim
        + cfg.linear_num_value_heads * cfg.linear_value_head_dim
    )
    layer.recurrent_states[0] = recurrent
    layer.conv_states[0] = torch.zeros(
        batch,
        conv_dim,
        cfg.linear_conv_kernel_dim,
        dtype=model.dtype,
        device=device,
    )
    layer.conv_kernel_size[0] = cfg.linear_conv_kernel_dim
    layer.is_conv_states_initialized[0] = True
    layer.is_recurrent_states_initialized[0] = True
    layer.has_previous_state[0] = True
    layer.dtype, layer.device = model.dtype, device
    return cache


def save_grid(saved: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(saved, temporary)
    temporary.replace(path)


def visible_token_ids(
    generated: Iterable[int],
    protocol: Protocol,
    tool_start: int,
    tool_end: int,
    im_end: int,
) -> tuple[list[int], bool]:
    """Remove private transport and native tool envelopes for presentation only.

    Operation names and arguments are never decoded here.  The two tool IDs
    merely delimit Qwen's internal message to the compiled executor.
    """
    visible: list[int] = []
    in_call = False
    saw_call = False
    for raw in generated:
        token = int(raw)
        if protocol.is_private(token):
            continue
        if token == tool_start:
            in_call = True
            saw_call = True
            continue
        if in_call:
            if token == tool_end:
                in_call = False
            continue
        if token != im_end:
            visible.append(token)
    return visible, saw_call


class InternalRuntime:
    def __init__(
        self,
        model_path: Path,
        state_path: Path,
        neural_path: Path,
        device: str,
        max_new_tokens: int = 512,
    ) -> None:
        self.model_path = Path(model_path)
        self.state_path = Path(state_path)
        self.neural_path = Path(neural_path)
        self.protocol = Protocol.load(self.model_path)
        self.max_new_tokens = max(32, int(max_new_tokens))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = Qwen3_5ForCausalLM.from_pretrained(
            self.model_path, dtype=torch.bfloat16
        ).to(device).eval()
        self.im_end = self._special_id("<|im_end|>")
        self.tool_start = self._special_id("<tool_call>")
        self.tool_end = self._special_id("</tool_call>")
        self.generation = 0
        self.event_id = 0
        self.conversation_id = 0
        self.conversation_epoch = 0
        self.conversation: list[dict[str, Any]] = []
        self.cache = self._load_or_initialize()
        self.publish("READY", "Internal neural controller loaded")

    def _special_id(self, value: str) -> int:
        token = self.tokenizer.convert_tokens_to_ids(value)
        if token is None or int(token) < 0:
            raise ValueError(f"tokenizer does not contain {value!r}")
        return int(token)

    @torch.no_grad()
    def _feed(self, tokens: list[int], cache=None):
        if not tokens:
            return cache, None
        tensor = torch.tensor([tokens], device=self.model.device)
        output = self.model(
            input_ids=tensor, past_key_values=cache, use_cache=True, logits_to_keep=1
        )
        return output.past_key_values, output.logits[0, -1]

    def _load_or_initialize(self):
        if self.neural_path.exists():
            saved = torch.load(self.neural_path, weights_only=True)
            cache = restore_grid(self.model, saved, self.protocol)
            if self.state_path.exists():
                old = json.loads(self.state_path.read_text(encoding="utf-8"))
                self.generation = int(old.get("generation", 0))
                self.event_id = int(old.get("event_id", 0))
                self.conversation_epoch = int(old.get("conversation_epoch", 0))
                items = old.get("conversation", [])
                if isinstance(items, list):
                    self.conversation = [x for x in items if isinstance(x, dict)][-100:]
                    self.conversation_id = max(
                        (int(x.get("id", 0)) for x in self.conversation), default=0
                    )
            return cache

        # RESET and the packed rows are protocol inputs, not host-side game
        # logic. Feed token by token so initialization exercises the identical
        # recurrent path used by generated internal rows.
        cache = None
        for token in [
            self.protocol.reset_token,
            *(self.protocol.row_base + int(row) for row in INITIAL_ROWS),
            self.protocol.done_token,
        ]:
            cache, _ = self._feed([token], cache)
        if read_rows(cache, self.protocol) != tuple(INITIAL_ROWS):
            raise AssertionError("directly programmed RESET did not initialize neural state")
        save_grid(snapshot_grid(cache, self.protocol), self.neural_path)
        return cache

    @property
    def rows(self) -> tuple[int, ...]:
        return read_rows(self.cache, self.protocol)

    def record(self, role: str, text: str, kind: str) -> None:
        self.conversation_id += 1
        self.conversation.append(
            {"id": self.conversation_id, "role": role, "text": str(text), "kind": kind}
        )
        self.conversation = self.conversation[-100:]

    def publish(self, parsed: str, message: str) -> None:
        self.event_id += 1
        atomic_json(
            self.state_path,
            state_document(
                self.rows,
                self.generation,
                parsed,
                message,
                self.event_id,
                self.conversation,
                self.conversation_epoch,
            ),
        )

    def _messages(self, user_text: str) -> list[dict[str, str]]:
        # The UI intentionally hides native tool envelopes.  Replaying those
        # visible acknowledgements to Qwen without the hidden tool call would
        # create a misleading few-shot example: "say it happened" rather than
        # call the internal executor.  Keep ordinary conversation in context,
        # but leave state-changing turns out; the authoritative continuity for
        # those turns is already carried by recurrent neural state.
        history = [
            {"role": str(item["role"]), "content": str(item["text"])}
            for item in self.conversation[-8:]
            if item.get("kind") in {"user-chat", "chat"}
            and item.get("role") in {"user", "assistant"}
        ]
        return [{"role": "system", "content": SYSTEM}, *history, {
            "role": "user",
            "content": user_text,
        }]

    @torch.no_grad()
    def turn(self, user_text: str) -> TurnResult:
        """Let Qwen converse or invoke its internal executor, transactionally."""
        before = snapshot_grid(self.cache, self.protocol)
        cache = restore_grid(self.model, before, self.protocol)
        encoded = self.tokenizer.apply_chat_template(
            self._messages(user_text),
            tools=TOOLS,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        )
        prompt = encoded["input_ids"].to(self.model.device)
        output = self.model(
            input_ids=prompt, past_key_values=cache, use_cache=True, logits_to_keep=1
        )
        cache = output.past_key_values
        generated: list[int] = []
        completed = False
        for _ in range(self.max_new_tokens):
            token = int(output.logits[0, -1].float().argmax())
            generated.append(token)
            if token == self.im_end:
                completed = True
                break
            current = torch.tensor([[token]], device=self.model.device)
            output = self.model(
                input_ids=current, past_key_values=cache, use_cache=True, logits_to_keep=1
            )
            cache = output.past_key_values
        if not completed:
            self.cache = restore_grid(self.model, before, self.protocol)
            raise RuntimeError("model turn exceeded its token budget; neural transaction rolled back")

        private = tuple(token for token in generated if self.protocol.is_private(token))
        rows_written = sum(self.protocol.is_row(token) for token in private)
        if rows_written and (
            rows_written % 16 != 0 or self.protocol.done_token not in private
        ):
            self.cache = restore_grid(self.model, before, self.protocol)
            raise RuntimeError("incomplete internal execution trace; neural transaction rolled back")

        # Only a complete private row trace is an authoritative neural-memory
        # transaction.  Ordinary text and rejected/malformed tool calls can
        # exercise the same recurrent layer while Qwen is decoding, but they
        # are not allowed to commit incidental numerical changes to the world.
        # This is protocol-level transaction handling, not semantic parsing.
        self.cache = cache if rows_written else restore_grid(self.model, before, self.protocol)
        save_grid(snapshot_grid(self.cache, self.protocol), self.neural_path)
        shown, saw_call = visible_token_ids(
            generated,
            self.protocol,
            self.tool_start,
            self.tool_end,
            self.im_end,
        )
        visible = self.tokenizer.decode(shown, skip_special_tokens=True).strip()
        commits = rows_written // 16
        if not visible:
            if commits:
                visible = "I applied that through my internal neural circuit."
            elif saw_call:
                visible = "I could not complete that internal operation."
            else:
                visible = "I'm here with the world held in my recurrent neural state."
        return TurnResult(
            visible=visible,
            generated=tuple(generated),
            private=private,
            tool_call_seen=saw_call,
            row_commits=commits,
            tick_tokens=sum(token == self.protocol.tick_token for token in private),
        )

    def submit(self, value: dict[str, Any]) -> TurnResult:
        if value.get("type") != "say":
            raise ValueError("direct controls are disabled; talk to Qwen to change the world")
        text = str(value.get("text", "")).strip()
        if not text:
            raise ValueError("message is empty")
        reset_chat = value.get("reset_chat", False)
        if type(reset_chat) is not bool:
            raise ValueError("reset_chat must be a boolean")
        record_chat = value.get("record_chat", True)
        if type(record_chat) is not bool:
            raise ValueError("record_chat must be a boolean")
        if reset_chat:
            # UI session control, not a host-side Life operation. The board
            # still changes only when Qwen invokes the programmed circuit.
            self.conversation.clear()
            self.conversation_epoch += 1
        result = self.turn(text)
        # The programmed protocol emits TICK only for actual Life transitions;
        # edits may commit rows but do not advance this display counter.
        self.generation += result.tick_tokens
        kind = "internal" if result.changed_through_circuit else "chat"
        if record_chat and not reset_chat:
            self.record("user", text, "user-internal" if kind == "internal" else "user-chat")
            self.record("assistant", result.visible, kind)
        parsed = "INTERNAL" if result.changed_through_circuit else "CHAT"
        self.publish(parsed, result.visible)
        return result


def process_queued_command(runtime, command: dict[str, Any]) -> str | None:
    try:
        runtime.submit(command)
        return None
    except Exception as error:
        message = f"Command rejected: {type(error).__name__}: {error}"
        if (command.get("type") == "say" and command.get("record_chat", True)
                and hasattr(runtime, "record")):
            runtime.record("assistant", message, "error")
        runtime.publish("REJECTED", message)
        return message


def consume_records(path: Path, offset: int) -> tuple[list[tuple[dict[str, Any], int]], int]:
    if not path.exists():
        return [], offset
    records: list[tuple[dict[str, Any], int]] = []
    with path.open("rb") as handle:
        handle.seek(offset)
        while line := handle.readline():
            offset = handle.tell()
            if line.strip():
                records.append((json.loads(line), offset))
    return records, offset


def prepare_queue(directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    commands = directory / "commands.jsonl"
    cursor = directory / "commands.offset"
    commands.touch(exist_ok=True)
    commands.chmod(0o666)
    return commands, cursor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--poll", type=float, default=0.05)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    directory = args.runtime_dir
    commands_path, cursor_path = prepare_queue(directory)
    runtime = InternalRuntime(
        args.model,
        directory / "state.json",
        directory / "neural_state.pt",
        args.device,
        args.max_new_tokens,
    )
    offset = int(cursor_path.read_text()) if cursor_path.exists() else 0
    print(f"Internal neural Life runtime watching {commands_path}", flush=True)
    try:
        while True:
            try:
                records, _ = consume_records(commands_path, offset)
                for command, committed in records:
                    process_queued_command(runtime, command)
                    offset = committed
                    cursor_path.write_text(str(offset), encoding="utf-8")
            except Exception as error:
                runtime.publish("ERROR", f"Runtime failure: {type(error).__name__}: {error}")
            time.sleep(args.poll)
    except KeyboardInterrupt:
        runtime.publish("STOPPED", "Internal neural runtime stopped")


if __name__ == "__main__":
    main()
