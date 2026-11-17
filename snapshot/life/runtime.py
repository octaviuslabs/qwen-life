"""End-to-end bridge between the merged checkpoint and the rule-free web UI.

The model cache is authoritative.  This process is allowed to clock the model,
load user-edited grids with RESET, and serialize the recurrent grid for the UI.
It never computes a Conway transition itself.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from .commands import (
    apply_input,
    canonical,
    is_read_only_conversation,
    parse_canonical,
    parse_user_intent,
    resolve_implicit_placement,
    rows_to_cells,
)
from .host import Feeder, load_model
from .monitor import load as load_neural
from .monitor import read, restore, save as save_neural, snapshot
from .reference import INITIAL_ROWS


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME = PROJECT / "ui" / "runtime"

PARSER_SYSTEM = """You are a strict command parser for a 16 by 16 Conway's Game of Life board.
Return exactly one line and no explanation, using one of these forms:
TICK n
CLEAR
PLACE GLIDER row column
PLACE BLINKER row column
PLACE BLOCK row column
SET CELL row column alive
INSPECT
Rows and columns are zero based, 0 through 15. alive is 0 or 1.
Return one command only. Never combine actions or guess missing coordinates.
Examples:
advance three generations -> TICK 3
put a glider at row 2 column 4 -> PLACE GLIDER 2 4
turn on cell at row 7 column 9 -> SET CELL 7 9 1
turn off cell at row 7 column 9 -> SET CELL 7 9 0
inspect the board -> INSPECT
erase everything -> CLEAR"""


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def state_document(
    rows: tuple[int, ...],
    generation: int,
    parsed_command: str = "",
    message: str = "Neural state ready",
    event_id: int = 0,
    conversation: list[dict[str, Any]] | None = None,
    conversation_epoch: int = 0,
) -> dict[str, Any]:
    cells = rows_to_cells(rows)
    return {
        "width": 16,
        "height": 16,
        "cells": cells,
        "generation": int(generation),
        "live_cells": sum(cells),
        "backend": "qwen-life-neural-cache",
        "parsed_command": parsed_command,
        "message": message,
        "event_id": int(event_id),
        "conversation": list(conversation or []),
        "conversation_epoch": int(conversation_epoch),
    }


class TextCommandParser:
    """Use the text half of the same merged checkpoint in an isolated cache."""

    def __init__(self, model, tokenizer, base_vocab: int, max_new_tokens: int = 40):
        self.model = model
        self.tokenizer = tokenizer
        self.base_vocab = int(base_vocab)
        self.max_new_tokens = max_new_tokens
        self.history: list[dict[str, str]] = []
        self.history_world: tuple[int, tuple[int, ...]] | None = None
        eos = tokenizer.eos_token_id
        self.eos = {int(eos)} if isinstance(eos, int) else {int(value) for value in eos or []}

    @torch.no_grad()
    def _generate(self, messages: list[dict[str, str]], limit: int) -> str:
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        ids = encoded["input_ids"].to(next(self.model.parameters()).device)
        out = self.model(input_ids=ids, use_cache=True)
        cache = out.past_key_values
        generated: list[int] = []
        for _ in range(limit):
            # Appended Life rows are tied output rows.  They are intentionally
            # excluded while Qwen is acting as the language parser.
            token = int(out.logits[0, -1, : self.base_vocab].float().argmax())
            if token in self.eos:
                break
            generated.append(token)
            current = torch.tensor([[token]], device=ids.device)
            out = self.model(input_ids=current, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def raw(self, text: str) -> str:
        return self._generate(
            [
                {"role": "system", "content": PARSER_SYSTEM},
                {"role": "user", "content": text},
            ],
            self.max_new_tokens,
        )

    def respond(self, text: str, rows: tuple[int, ...], generation: int) -> str:
        """Speak as the current world without granting this path write access."""
        world = (int(generation), tuple(rows))
        if self.history_world != world:
            # Prior replies may contain exact generation/cell facts. Once the
            # world changes they are stale context, so start a fresh dialogue.
            self.history.clear()
            self.history_world = world
        grid = "\n".join(
            "".join("#" if row >> x & 1 else "." for x in range(16))
            for row in rows
        )
        system = f"""You are the friendly voice of a living 16 by 16 Conway's Game of Life world.
Speak in first person as the world shown beside this conversation. Be concise and natural.
This conversation route is read-only: never claim that you changed the board.
If asked about the world, use these authoritative current facts and do not guess:
{describe_rows(rows, generation)}
Rows run top to bottom and columns left to right, both zero based. # is alive and . is dead.
{grid}"""
        messages = [{"role": "system", "content": system}, *self.history[-8:]]
        messages.append({"role": "user", "content": text})
        answer = self._generate(messages, 96).strip()
        if not answer:
            answer = "I'm here, quietly holding the current world in my neural state."
        self.history.extend(
            [
                {"role": "user", "content": text},
                {"role": "assistant", "content": answer},
            ]
        )
        self.history = self.history[-8:]
        return answer

    def parse(self, text: str) -> tuple[dict, str, bool]:
        # The model is a convenient semantic translator, not the authority to
        # mutate state. The user's own text must first map to exactly one safe
        # supported intent. Agreement is recorded as the normal model path;
        # disagreement uses the narrow guarded intent and is visibly labeled.
        guarded = parse_user_intent(text)
        raw = self.raw(text)
        try:
            translated = parse_canonical(raw)
        except ValueError:
            return guarded, raw, True
        if canonical(translated) != canonical(guarded):
            return guarded, raw, True
        return translated, raw, False


def describe_rows(rows: tuple[int, ...], generation: int) -> str:
    """Non-mutating description read directly from authoritative neural rows."""
    live = [(y, x) for y, row in enumerate(rows) for x in range(16) if row >> x & 1]
    if not live:
        return f"Generation {generation}: the neural board is empty (0 live cells)."
    ys, xs = [point[0] for point in live], [point[1] for point in live]
    return (
        f"Generation {generation}: {len(live)} live cells; occupied rows "
        f"{min(ys)}–{max(ys)} and columns {min(xs)}–{max(xs)}."
    )


def resolve_text_request(parser, text: str, rows: tuple[int, ...], generation: int) -> dict:
    """Resolve text to either a guarded command or a read-only world reply."""
    implicit = resolve_implicit_placement(text)
    if implicit is not None:
        command, note = implicit
        return {
            "kind": "command",
            "command": command,
            "raw": "",
            "fallback": False,
            "note": note,
        }
    try:
        command, raw, fallback = parser.parse(text)
    except ValueError:
        if not is_read_only_conversation(text):
            raise
        return {
            "kind": "chat",
            "message": parser.respond(text, rows, generation),
        }
    return {
        "kind": "command",
        "command": command,
        "raw": raw,
        "fallback": fallback,
    }


class NeuralRuntime:
    def __init__(
        self,
        model_path: Path,
        state_path: Path,
        neural_path: Path,
        device: str,
        recycle_every: int = 256,
    ) -> None:
        self.model_path = model_path
        self.state_path = state_path
        self.neural_path = neural_path
        self.recycle_every = max(1, int(recycle_every))
        self.model, self.offset = load_model(str(model_path), device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.text_parser = TextCommandParser(self.model, self.tokenizer, self.offset)
        self.generation = 0
        self.event_id = 0
        self.conversation_id = 0
        self.conversation_epoch = 0
        self.conversation: list[dict[str, Any]] = []
        self.since_recycle = 0
        self.feeder = self._start()
        self.publish(message="Neural runtime loaded")

    def _start(self) -> Feeder:
        if self.neural_path.exists():
            saved = load_neural(self.neural_path)
            feeder = Feeder(self.model, self.offset, cache=restore(self.model, saved))
            if self.state_path.exists():
                previous = json.loads(self.state_path.read_text(encoding="utf-8"))
                self.generation = int(previous.get("generation", 0))
                self.event_id = int(previous.get("event_id", 0))
                self.conversation_epoch = int(previous.get("conversation_epoch", 0))
                saved_conversation = previous.get("conversation", [])
                if isinstance(saved_conversation, list):
                    self.conversation = [
                        item for item in saved_conversation if isinstance(item, dict)
                    ][-100:]
                    self.conversation_id = max(
                        (int(item.get("id", 0)) for item in self.conversation),
                        default=0,
                    )
            return feeder
        feeder = Feeder(self.model, self.offset)
        feeder.reset(INITIAL_ROWS)
        self._save_cache(feeder)
        return feeder

    def _save_cache(self, feeder: Feeder | None = None) -> None:
        feeder = feeder or self.feeder
        self.neural_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.neural_path.with_name(f".{self.neural_path.name}.{os.getpid()}.tmp")
        save_neural(snapshot(feeder.cache), temporary)
        temporary.replace(self.neural_path)

    @property
    def rows(self) -> tuple[int, ...]:
        return read(self.feeder.cache)

    def publish(self, parsed: str = "", message: str = "Neural state committed") -> None:
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

    def record_conversation(self, role: str, text: str, kind: str = "") -> None:
        self.conversation_id += 1
        self.conversation.append(
            {
                "id": self.conversation_id,
                "role": role,
                "text": str(text),
                "kind": kind,
            }
        )
        self.conversation = self.conversation[-100:]

    def reset_rows(self, rows: tuple[int, ...]) -> None:
        # A fresh cache prevents irrelevant native attention history from
        # growing when a user edits the world.
        feeder = Feeder(self.model, self.offset)
        feeder.reset(rows)
        if read(feeder.cache) != tuple(rows):
            raise AssertionError("RESET rows did not reach neural state")
        self.feeder = feeder
        self.since_recycle = 0

    def recycle(self) -> None:
        saved = snapshot(self.feeder.cache)
        self.feeder = Feeder(self.model, self.offset, cache=restore(self.model, saved))
        self.since_recycle = 0

    def execute(self, command: dict, *, parsed: str | None = None, note: str = "") -> None:
        kind = command["type"]
        if kind == "tick":
            for _ in range(int(command["count"])):
                emitted = self.feeder.generation()
                state = self.rows
                if emitted != state:
                    raise AssertionError("emitted rows disagree with direct neural-state readout")
                self.generation += 1
                self.since_recycle += 1
                if self.since_recycle >= self.recycle_every:
                    self.recycle()
        elif kind in {"clear", "place", "set_cell"}:
            self.reset_rows(apply_input(self.rows, command))
            if kind == "clear":
                self.generation = 0
        elif kind == "inspect":
            rendered = parsed or canonical(command)
            description = describe_rows(self.rows, self.generation)
            message = f"{note}. {description}" if note else description
            self.publish(rendered, message)
            return
        else:
            raise ValueError(f"unsupported executable command: {kind!r}")
        self._save_cache()
        rendered = parsed or canonical(command)
        self.publish(rendered, note or "Command executed by the neural runtime")

    def submit(self, value: dict[str, Any]) -> None:
        record_chat = value.get("record_chat", True)
        if type(record_chat) is not bool:
            raise ValueError("record_chat must be a boolean")
        if "record_chat" in value and value.get("type") != "say":
            raise ValueError("record_chat is only supported for say")
        reset_chat = value.get("reset_chat", False)
        if type(reset_chat) is not bool:
            raise ValueError("reset_chat must be a boolean")
        if "reset_chat" in value and value.get("type") not in {"clear", "say"}:
            raise ValueError("reset_chat is only supported for clear and say")
        if reset_chat:
            self.conversation.clear()
            self.conversation_epoch += 1
            self.text_parser.history.clear()
            self.text_parser.history_world = None
        if value.get("type") != "say":
            command = parse_canonical(canonical(value))
            self.execute(command)
            return
        text = str(value.get("text", "")).strip()
        if not text:
            raise ValueError("natural-language command is empty")
        if record_chat and not reset_chat:
            self.record_conversation("user", text)
        if not record_chat:
            previous_history = list(self.text_parser.history)
            previous_world = self.text_parser.history_world
        try:
            resolved = resolve_text_request(self.text_parser, text, self.rows, self.generation)
        finally:
            if not record_chat:
                self.text_parser.history = previous_history
                self.text_parser.history_world = previous_world
        if reset_chat:
            # A conversational answer can itself add parser history. The reset
            # action and its reply must not seed the new conversation.
            self.text_parser.history.clear()
            self.text_parser.history_world = None
        if resolved["kind"] == "chat":
            if record_chat and not reset_chat:
                self.record_conversation("assistant", resolved["message"])
            self.publish(parsed="CHAT", message=resolved["message"])
            return
        command, raw, fallback = (
            resolved["command"],
            resolved["raw"],
            resolved["fallback"],
        )
        note = resolved.get("note") or (
            f"Qwen output {raw!r} disagreed; executed guarded {canonical(command)}"
            if fallback
            else f"Qwen parsed the request as {canonical(command)}"
        )
        reply = (
            f"{note}. {describe_rows(self.rows, self.generation)}"
            if command["type"] == "inspect"
            else note
        )
        if record_chat and not reset_chat:
            self.record_conversation("assistant", reply)
        self.execute(command, parsed=canonical(command), note=note)


def process_queued_command(runtime, command: dict[str, Any]) -> str | None:
    """Execute one queue record; rejected records are surfaced and consumed."""
    try:
        runtime.submit(command)
        return None
    except Exception as error:
        message = f"Command rejected: {type(error).__name__}: {error}"
        if (
            command.get("type") == "say"
            and command.get("record_chat", True) is not False
            and hasattr(runtime, "record_conversation")
        ):
            runtime.record_conversation("assistant", message, "error")
        runtime.publish(parsed="REJECTED", message=message)
        return message


def consume(path: Path, offset: int) -> tuple[list[dict[str, Any]], int]:
    records, offset = consume_records(path, offset)
    return [command for command, _ in records], offset


def consume_records(path: Path, offset: int) -> tuple[list[tuple[dict[str, Any], int]], int]:
    if not path.exists():
        return [], offset
    commands: list[tuple[dict[str, Any], int]] = []
    with path.open("rb") as handle:
        handle.seek(offset)
        while line := handle.readline():
            offset = handle.tell()
            if line.strip():
                commands.append((json.loads(line), offset))
    return commands, offset


def prepare_queue(directory: Path) -> tuple[Path, Path]:
    """Create the local Docker-to-host handoff before either side writes it.

    The checked-in launch command runs the neural process as container root and
    the HTTP bridge as the desktop user.  The queue contains commands only and
    is intentionally writable by both local processes.
    """
    directory.mkdir(parents=True, exist_ok=True)
    commands_path = directory / "commands.jsonl"
    cursor_path = directory / "commands.offset"
    commands_path.touch(exist_ok=True)
    commands_path.chmod(0o666)
    return commands_path, cursor_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--poll", type=float, default=0.05)
    parser.add_argument("--recycle-every", type=int, default=256)
    args = parser.parse_args()
    directory = args.runtime_dir
    commands_path, cursor_path = prepare_queue(directory)
    runtime = NeuralRuntime(
        args.model,
        directory / "state.json",
        directory / "neural_state.pt",
        args.device,
        args.recycle_every,
    )
    offset = int(cursor_path.read_text()) if cursor_path.exists() else 0
    print(f"Neural Life runtime watching {commands_path}", flush=True)
    try:
        while True:
            try:
                records, _ = consume_records(commands_path, offset)
                for command, committed_offset in records:
                    process_queued_command(runtime, command)
                    offset = committed_offset
                    cursor_path.write_text(str(offset), encoding="utf-8")
            except Exception as error:  # surface errors in the UI, keep the demo alive
                runtime.publish(message=f"Command failed: {type(error).__name__}: {error}")
            time.sleep(args.poll)
    except KeyboardInterrupt:
        runtime.publish(message="Neural runtime stopped")


if __name__ == "__main__":
    main()
