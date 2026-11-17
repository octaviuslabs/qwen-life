"""A rule-free web bridge for the neural Game of Life renderer.

This process never computes a Life generation.  It serves snapshots supplied by
the model runtime and forwards structured commands back to that runtime.  The
``--mock`` backend only cycles through a few recorded frames so the interface
can be developed before the Qwen integration is ready.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import mimetypes
import threading
from dataclasses import dataclass, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
GRID_SIZE = 16


@dataclass(frozen=True)
class Snapshot:
    width: int
    height: int
    cells: tuple[int, ...]
    generation: int = 0
    live_cells: int = 0
    backend: str = "waiting-for-neural-runtime"
    parsed_command: str = ""
    message: str = ""
    event_id: int = 0
    conversation: tuple[dict[str, Any], ...] = ()
    conversation_epoch: int = 0
    completed_offset: int = 0
    queue_pending: bool = False

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("snapshot dimensions must be positive")
        if len(self.cells) != self.width * self.height:
            raise ValueError("cell count does not match snapshot dimensions")
        if any(cell not in (0, 1) for cell in self.cells):
            raise ValueError("snapshot cells must be binary")
        if min(self.generation, self.live_cells, self.event_id, self.conversation_epoch,
               self.completed_offset) < 0:
            raise ValueError("snapshot counters must be non-negative")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Snapshot":
        cells = tuple(int(cell) for cell in value["cells"])
        return cls(
            width=int(value["width"]),
            height=int(value["height"]),
            cells=cells,
            generation=int(value.get("generation", 0)),
            live_cells=int(value.get("live_cells", sum(cells))),
            backend=str(value.get("backend", "neural-runtime")),
            parsed_command=str(value.get("parsed_command", "")),
            message=str(value.get("message", "")),
            event_id=int(value.get("event_id", 0)),
            conversation=tuple(
                item for item in value.get("conversation", []) if isinstance(item, dict)
            ),
            conversation_epoch=int(value.get("conversation_epoch", 0)),
            completed_offset=int(value.get("completed_offset", 0)),
            queue_pending=bool(value.get("queue_pending", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "cells": list(self.cells),
            "generation": self.generation,
            "live_cells": self.live_cells,
            "backend": self.backend,
            "parsed_command": self.parsed_command,
            "message": self.message,
            "event_id": self.event_id,
            "conversation": [dict(item) for item in self.conversation],
            "conversation_epoch": self.conversation_epoch,
            "completed_offset": self.completed_offset,
            "queue_pending": self.queue_pending,
        }


class Backend(Protocol):
    """The only contract between the UI and a state-producing runtime."""

    def snapshot(self) -> Snapshot: ...

    def submit(self, command: dict[str, Any]) -> dict[str, Any]: ...


def _blank_snapshot(message: str = "Waiting for the neural runtime") -> Snapshot:
    return Snapshot(
        GRID_SIZE,
        GRID_SIZE,
        (0,) * (GRID_SIZE * GRID_SIZE),
        backend="waiting-for-neural-runtime",
        message=message,
    )


class FileBridgeBackend:
    """Read snapshots and append commands without interpreting either one.

    The Qwen runner atomically replaces ``state_path`` after it changes neural
    state.  This bridge appends one JSON command per line to ``commands_path``.
    It does not derive a next state from a command.
    """

    def __init__(self, state_path: Path, commands_path: Path) -> None:
        self.state_path = state_path
        self.commands_path = commands_path
        self._lock = threading.Lock()

    def snapshot(self) -> Snapshot:
        # The runtime publishes state before advancing its queue cursor. Read
        # that cursor first so a completion receipt never accompanies an older
        # board. A cursor read during its brief truncate/write window is safely
        # conservative: the next poll will observe the committed offset.
        try:
            completed_offset = int(self.commands_path.with_suffix(".offset").read_text() or "0")
        except (FileNotFoundError, ValueError):
            completed_offset = 0
        try:
            value = json.loads(self.state_path.read_text())
        except FileNotFoundError:
            snapshot = _blank_snapshot(f"Waiting for {self.state_path.name}")
        else:
            snapshot = Snapshot.from_dict(value)
        try:
            queue_size = self.commands_path.stat().st_size
        except FileNotFoundError:
            queue_size = 0
        return replace(snapshot, completed_offset=completed_offset,
                       queue_pending=queue_size > completed_offset)

    def submit(self, command: dict[str, Any]) -> dict[str, Any]:
        normalized = validate_command(command)
        payload = (json.dumps(normalized, separators=(",", ":")) + "\n").encode("utf-8")
        self.commands_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.commands_path.open("ab") as handle:
            # Both tailnet UI processes share this queue. Lock the complete
            # append/flush/tell operation to return this request's own end,
            # rather than a later command's position from a concurrent writer.
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.write(payload)
            handle.flush()
            queue_offset = handle.tell()
        return {
            "accepted": True,
            "parsed_command": format_command(normalized),
            "message": "Queued for the neural runtime",
            "queue_offset": queue_offset,
        }


class NaturalLanguageControllerBackend:
    """Make Qwen the controller for every UI action.

    The browser's buttons use a small structural protocol because coordinates
    and toggle state are already known there.  The directly programmed
    checkpoint intentionally accepts only natural-language ``say`` records,
    however: Qwen must choose the internal tool and its arguments before the
    analytic circuit can execute it.  This adapter expresses each validated UI
    action as an unambiguous sentence and forwards that sentence to the normal
    bridge.  It never executes or parses a Life operation itself.

    Keeping this as an opt-in wrapper preserves the original checkpoint demo,
    whose runtime still accepts the structural records directly.
    """

    def __init__(self, backend: Backend) -> None:
        self.backend = backend

    def snapshot(self) -> Snapshot:
        return self.backend.snapshot()

    def submit(self, command: dict[str, Any]) -> dict[str, Any]:
        normalized = validate_command(command)
        return self.backend.submit(command_as_natural_language(normalized))


MOCK_FRAMES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((1, 2), (2, 3), (3, 1), (3, 2), (3, 3)),
    ((2, 1), (2, 3), (3, 2), (3, 3), (4, 2)),
    ((2, 3), (3, 1), (3, 3), (4, 2), (4, 3)),
    ((2, 2), (3, 3), (3, 4), (4, 2), (4, 3)),
)


def _fixture_cells(points: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    cells = [0] * (GRID_SIZE * GRID_SIZE)
    for row, column in points:
        cells[row * GRID_SIZE + column] = 1
    return tuple(cells)


class MockFixtureBackend:
    """Recorded UI fixtures, not a Game of Life implementation."""

    def __init__(self) -> None:
        self._index = 0
        self._last_command = ""
        self._conversation_epoch = 0
        self._lock = threading.Lock()

    def snapshot(self) -> Snapshot:
        with self._lock:
            cells = _fixture_cells(MOCK_FRAMES[self._index])
            return Snapshot(
                GRID_SIZE,
                GRID_SIZE,
                cells,
                generation=self._index,
                live_cells=sum(cells),
                backend="mock-recorded-frames",
                parsed_command=self._last_command,
                message="UI preview only; no simulation is running",
                conversation_epoch=self._conversation_epoch,
            )

    def submit(self, command: dict[str, Any]) -> dict[str, Any]:
        normalized = validate_command(command)
        rendered = format_command(normalized)
        with self._lock:
            if normalized.get("reset_chat", False):
                self._conversation_epoch += 1
            if normalized["type"] == "tick":
                self._index = (self._index + int(normalized.get("count", 1))) % len(MOCK_FRAMES)
            elif normalized["type"] == "clear":
                self._index = 0
            self._last_command = rendered
        return {
            "accepted": True,
            "parsed_command": rendered,
            "message": "Mock mode recorded the command; it did not execute Life rules",
        }


COMMAND_TYPES = {"tick", "clear", "place", "set_cell", "say"}
PATTERN_EXTENTS = {"glider": (2, 2), "blinker": (0, 2), "block": (1, 1)}


def validate_command(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("command must be an object")
    command_type = value.get("type")
    if command_type not in COMMAND_TYPES:
        raise ValueError(f"unsupported command type: {command_type!r}")
    allowed = {
        "tick": {"type", "count"},
        "clear": {"type", "reset_chat"},
        "place": {"type", "pattern", "row", "column"},
        "set_cell": {"type", "row", "column", "alive"},
        "say": {"type", "text", "reset_chat", "record_chat"},
    }[command_type]
    unexpected = set(value) - allowed
    if unexpected:
        raise ValueError(f"unexpected command fields: {sorted(unexpected)}")

    def integer(name: str, default: int | None = None) -> int:
        if name not in value:
            if default is None:
                raise ValueError(f"missing integer field: {name}")
            return default
        result = value[name]
        # JSON booleans are Python ints, but they are not valid protocol values.
        if type(result) is not int:
            raise ValueError(f"{name} must be an integer")
        return result

    normalized: dict[str, Any] = {"type": command_type}
    for flag in ("reset_chat", "record_chat"):
        if flag in value:
            if type(value[flag]) is not bool:
                raise ValueError(f"{flag} must be a boolean")
            normalized[flag] = value[flag]
    if command_type == "tick":
        count = integer("count", 1)
        if not 1 <= count <= 10_000:
            raise ValueError("tick count must be between 1 and 10000")
        normalized["count"] = count
    elif command_type == "place":
        if type(value.get("pattern")) is not str:
            raise ValueError("pattern must be a string")
        pattern = value["pattern"].lower()
        if pattern not in PATTERN_EXTENTS:
            raise ValueError("unknown pattern")
        row, column = integer("row", 1), integer("column", 1)
        height, width = PATTERN_EXTENTS[pattern]
        if row < 0 or column < 0 or row + height >= GRID_SIZE or column + width >= GRID_SIZE:
            raise ValueError("pattern is outside the 16 by 16 grid")
        normalized.update(
            pattern=pattern,
            row=row,
            column=column,
        )
    elif command_type == "set_cell":
        row, column, alive = integer("row"), integer("column"), integer("alive")
        if not 0 <= row < GRID_SIZE or not 0 <= column < GRID_SIZE:
            raise ValueError("cell is outside the 16 by 16 grid")
        if alive not in (0, 1):
            raise ValueError("alive must be zero or one")
        normalized.update(row=row, column=column, alive=alive)
    elif command_type == "say":
        if type(value.get("text")) is not str:
            raise ValueError("text must be a string")
        text = value["text"].strip()
        if not text or len(text) > 2_000:
            raise ValueError("text must contain between 1 and 2000 characters")
        normalized["text"] = text
    return normalized


def format_command(command: dict[str, Any]) -> str:
    kind = command["type"]
    if kind == "tick":
        return f"TICK {command['count']}"
    if kind == "clear":
        return "CLEAR"
    if kind == "place":
        return f"PLACE {command['pattern'].upper()} {command['row']} {command['column']}"
    if kind == "set_cell":
        return f"SET CELL {command['row']} {command['column']} {command['alive']}"
    return f"SAY {command['text']}"


def command_as_natural_language(command: dict[str, Any]) -> dict[str, Any]:
    """Express one validated UI command as language for Qwen to interpret."""
    kind = command["type"]
    if kind == "say":
        text = command["text"]
    elif kind == "tick":
        count = command["count"]
        text = (
            "Advance one generation."
            if count == 1
            else f"Advance exactly {count} generations."
        )
    elif kind == "clear":
        text = "Clear the entire world now."
    elif kind == "place":
        text = (
            f"Place a {command['pattern']} with its origin at row "
            f"{command['row']} column {command['column']}."
        )
    elif kind == "set_cell":
        state = "alive" if command["alive"] else "dead"
        text = (
            f"Make the cell at row {command['row']} column "
            f"{command['column']} {state}."
        )
    else:  # validate_command makes this unreachable.
        raise ValueError(f"unsupported command type: {kind!r}")
    result: dict[str, Any] = {"type": "say", "text": text}
    if kind != "say":
        result["record_chat"] = False
    elif "record_chat" in command:
        result["record_chat"] = command["record_chat"]
    if "reset_chat" in command:
        result["reset_chat"] = command["reset_chat"]
    return result


class AppServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], backend: Backend) -> None:
        super().__init__(address, RequestHandler)
        self.backend = backend


class RequestHandler(BaseHTTPRequestHandler):
    server: AppServer

    def log_message(self, format: str, *args: object) -> None:
        return

    def _json(self, value: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/state":
            self._json(self.server.backend.snapshot().to_dict())
            return
        if path == "/api/health":
            self._json({"ok": True})
            return
        relative = "index.html" if path == "/" else path.lstrip("/")
        target = (STATIC / relative).resolve()
        if STATIC.resolve() not in target.parents or not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        payload = target.read_bytes()
        mime, _ = mimetypes.guess_type(target.name)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{mime or 'application/octet-stream'}; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/command":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size > 16_384:
                raise ValueError("request is too large")
            value = json.loads(self.rfile.read(size) or b"{}")
            result = self.server.backend.submit(value)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            self._json({"accepted": False, "error": str(error)}, HTTPStatus.BAD_REQUEST)
            return
        self._json(result, HTTPStatus.ACCEPTED)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--mock", action="store_true", help="cycle recorded UI fixtures")
    parser.add_argument("--state", type=Path, default=ROOT / "runtime" / "state.json")
    parser.add_argument("--commands", type=Path, default=ROOT / "runtime" / "commands.jsonl")
    parser.add_argument(
        "--internal-controller",
        action="store_true",
        help="send every UI action to the runtime as natural language for Qwen to route",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend: Backend
    if args.mock:
        backend = MockFixtureBackend()
        mode = "mock recorded frames"
    else:
        backend = FileBridgeBackend(args.state, args.commands)
        mode = f"neural bridge ({args.state})"
    if args.internal_controller:
        backend = NaturalLanguageControllerBackend(backend)
        mode += ", Qwen internal controller"
    server = AppServer((args.host, args.port), backend)
    print(f"Life UI on http://{args.host}:{args.port} — {mode}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
