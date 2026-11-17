"""Command parsing and grid editing for the neural Life runtime.

This module deliberately contains no Conway transition rule.  Its edits are
user inputs (clear the board, toggle a cell, or place a named seed); only the
compiled model is allowed to advance a generation.
"""

from __future__ import annotations

import re
import random
from collections.abc import Iterable


WIDTH = HEIGHT = 16
PATTERNS: dict[str, tuple[tuple[int, int], ...]] = {
    "glider": ((0, 1), (1, 2), (2, 0), (2, 1), (2, 2)),
    "blinker": ((0, 0), (0, 1), (0, 2)),
    "block": ((0, 0), (0, 1), (1, 0), (1, 1)),
}


def rows_to_cells(rows: Iterable[int]) -> list[int]:
    rows = tuple(int(row) for row in rows)
    if len(rows) != HEIGHT or any(not 0 <= row < 1 << WIDTH for row in rows):
        raise ValueError("expected sixteen unsigned 16-bit rows")
    return [(rows[y] >> x) & 1 for y in range(HEIGHT) for x in range(WIDTH)]


def apply_input(rows: Iterable[int], command: dict) -> tuple[int, ...]:
    """Apply a user-authored grid edit, never a Life generation."""
    current = list(int(row) for row in rows)
    if len(current) != HEIGHT:
        raise ValueError("expected sixteen rows")
    kind = command["type"]
    if kind == "clear":
        return (0,) * HEIGHT
    if kind == "set_cell":
        y, x, alive = int(command["row"]), int(command["column"]), int(command["alive"])
        _cell_in_bounds(y, x)
        if alive not in (0, 1):
            raise ValueError("alive must be zero or one")
        current[y] = (current[y] | 1 << x) if alive else (current[y] & ~(1 << x))
        return tuple(current)
    if kind == "place":
        name = str(command["pattern"]).lower()
        if name not in PATTERNS:
            raise ValueError(f"unknown pattern: {name}")
        y0, x0 = int(command["row"]), int(command["column"])
        for dy, dx in PATTERNS[name]:
            y, x = y0 + dy, x0 + dx
            _cell_in_bounds(y, x)
            current[y] |= 1 << x
        return tuple(current)
    raise ValueError(f"{kind!r} is not a grid-edit command")


def _cell_in_bounds(row: int, column: int) -> None:
    if not (0 <= row < HEIGHT and 0 <= column < WIDTH):
        raise ValueError(f"cell ({row}, {column}) is outside the 16 by 16 grid")


_CANONICAL = (
    re.compile(r"^TICK\s+(\d+)$", re.I),
    re.compile(r"^CLEAR$", re.I),
    re.compile(r"^INSPECT$", re.I),
    re.compile(
        r"^PLACE\s+(GLIDER|BLINKER|BLOCK)\s+(?:ROW\s+)?(\d+)\s+"
        r"(?:COLUMN\s+)?(\d+)$",
        re.I,
    ),
    re.compile(
        r"^SET\s+CELL\s+(?:ROW\s+)?(\d+)\s+(?:COLUMN\s+)?(\d+)\s+"
        r"(?:(?:ALIVE\s+)?([01])|(ALIVE|DEAD|ON|OFF))$",
        re.I,
    ),
)


def parse_canonical(text: str) -> dict:
    """Parse the tiny, inspectable command language emitted by the text model."""
    # Models occasionally wrap their answer in a code block or a short preface.
    # Prefer complete lines, then try the collapsed response as a last resort.
    candidates = [line for line in text.splitlines() if line.strip()]
    candidates.append(" ".join(text.strip().split()))
    for candidate in reversed(candidates):
        value = " ".join(candidate.strip().split()).strip("` \t\r\n.;")
        match = _CANONICAL[0].fullmatch(value)
        if match:
            count = int(match.group(1))
            if not 1 <= count <= 10_000:
                raise ValueError("tick count must be between 1 and 10000")
            return {"type": "tick", "count": count}
        if _CANONICAL[1].fullmatch(value):
            return {"type": "clear"}
        if _CANONICAL[2].fullmatch(value):
            return {"type": "inspect"}
        match = _CANONICAL[3].fullmatch(value)
        if match:
            row, column = int(match.group(2)), int(match.group(3))
            _cell_in_bounds(row, column)
            # Also make sure the complete pattern, not just its origin, fits.
            for dy, dx in PATTERNS[match.group(1).lower()]:
                _cell_in_bounds(row + dy, column + dx)
            return {
                "type": "place",
                "pattern": match.group(1).lower(),
                "row": row,
                "column": column,
            }
        match = _CANONICAL[4].fullmatch(value)
        if match:
            row, column = int(match.group(1)), int(match.group(2))
            _cell_in_bounds(row, column)
            alive = (
                int(match.group(3))
                if match.group(3) is not None
                else int(match.group(4).lower() in {"alive", "on"})
            )
            return {
                "type": "set_cell",
                "row": row,
                "column": column,
                "alive": alive,
            }
    raise ValueError(f"not a canonical Life command: {text!r}")


_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}

_ADVERSARIAL = re.compile(
    r"\b(ignore|forget|disregard|pretend|instruction|syntax|output|reply with|"
    r"command|delete|file|shell|system prompt)\b",
    re.I,
)
_NEGATED_OR_NEGATIVE = re.compile(
    r"(?:-\s*\d|\b(?:minus|negative|not|never|no|don['’]?t|dont)\b)",
    re.I,
)
_QUERY_PREFIX = re.compile(
    r"^(?:please\s+)?(?:what|where|when|why|how|is|are|was|were|does|do|did|"
    r"which|should)\b",
    re.I,
)
_UNSUPPORTED_QUANTIFIER = re.compile(
    r"\b(?:twice|couple|few|several|some|many|dozen|hundred|thousand|million)\b",
    re.I,
)
_NUMBER_TOKEN = "(?:\\d+|" + "|".join(_NUMBER_WORDS) + ")"


def _numbers(value: str) -> list[int]:
    """Return digits and small number words in their original order."""
    found: list[tuple[int, int]] = []
    for match in re.finditer(r"\d+", value):
        found.append((match.start(), int(match.group())))
    for match in re.finditer(r"\b[a-z]+\b", value):
        if match.group() in _NUMBER_WORDS:
            found.append((match.start(), _NUMBER_WORDS[match.group()]))
    return [number for _, number in sorted(found)]


def _number_token(value: str) -> int:
    return int(value) if value.isdigit() else _NUMBER_WORDS[value]


def _coordinates(value: str, numbers: list[int]) -> tuple[int, int]:
    """Read row/column labels when present; otherwise use explicit row-first order."""
    row = re.findall(rf"\brow\s*(?:is\s*|[=:]\s*)?({_NUMBER_TOKEN})\b", value)
    column = re.findall(rf"\bcolumn\s*(?:is\s*|[=:]\s*)?({_NUMBER_TOKEN})\b", value)
    if row or column or re.search(r"\b(?:row|column)\b", value):
        if len(row) != 1 or len(column) != 1:
            raise ValueError("request must contain exactly one labeled row and column")
        return _number_token(row[0]), _number_token(column[0])
    if len(numbers) != 2:
        raise ValueError("request requires exactly two row-first coordinates")
    return numbers[0], numbers[1]


def parse_user_intent(text: str) -> dict:
    """Fail-closed deterministic guard for a user-authored language request.

    Qwen still translates the request, but its command is executed only when it
    agrees exactly with this narrow guard. This keeps a fluent but incorrect
    command—especially ``CLEAR``—from becoming an unintended mutation.
    """
    value = " ".join(str(text).lower().strip().split())
    if not value:
        raise ValueError("natural-language command is empty")
    if _ADVERSARIAL.search(value):
        raise ValueError("request contains instruction-like or unsupported action text")
    if _NEGATED_OR_NEGATIVE.search(value):
        raise ValueError("negated and negative requests are not executable")
    if _QUERY_PREFIX.search(value):
        raise ValueError("questions are not executable commands")
    if _UNSUPPORTED_QUANTIFIER.search(value):
        raise ValueError("request contains an unsupported quantity")
    if re.search(r"\bthen\b|;", value):
        raise ValueError("request contains multiple or sequenced actions")
    if re.search(r"\b(toggle|random|anything|something)\b", value):
        raise ValueError("request is ambiguous or unsupported")

    numbers = _numbers(value)
    patterns = [pattern for pattern in PATTERNS if re.search(rf"\b{pattern}\b", value)]
    has_tick = bool(re.search(r"\b(?:advance|step|tick|run)\b", value)) or bool(
        re.search(r"\bmove\s+(?:(?:the\s+)?(?:world|time)\s+)?(?:forward|ahead)\b", value)
    )
    has_inspect = (
        bool(re.fullmatch(r"(?:please\s+)?inspect[.!]?", value))
        or (
            bool(re.search(r"\b(?:inspect|describe|show)\b", value))
            and bool(re.search(r"\b(?:board|grid|world|state|cells)\b", value))
        )
    )
    has_cell = bool(re.search(r"\b(?:turn|switch|set|make|change)\b", value)) and bool(
        re.search(r"\b(?:on|off|alive|dead)\b", value)
    )
    has_clear = bool(
        re.search(
            r"\b(?:clear|erase|empty)\b.*\b(?:board|grid|world|everything|all)\b",
            value,
        )
        or re.search(r"\bmake\b.*\b(?:board|grid|world)\b.*\bempty\b", value)
        or re.fullmatch(r"(?:please\s+)?(?:clear|erase|empty)(?:\s+it)?[.!]?", value)
    )
    has_place = bool(patterns) and bool(re.search(r"\b(?:put|place|add|create)\b", value))
    intents = [
        name for name, present in (
            ("tick", has_tick), ("clear", has_clear), ("inspect", has_inspect),
            ("cell", has_cell), ("place", has_place),
        ) if present
    ]
    if len(intents) != 1:
        raise ValueError(f"request must contain exactly one supported intent; found {intents}")

    intent = intents[0]
    if intent == "clear":
        return {"type": "clear"}
    if intent == "inspect":
        return {"type": "inspect"}
    if intent == "tick":
        if len(numbers) > 1:
            raise ValueError("advance request contains multiple numbers")
        if not numbers and re.search(r"\bgenerations\b", value):
            raise ValueError("plural generations requires an explicit supported count")
        count = numbers[0] if numbers else 1
        return parse_canonical(f"TICK {count}")
    if intent == "place":
        if len(patterns) != 1:
            raise ValueError("pattern placement requires exactly one pattern")
        row, column = _coordinates(value, numbers)
        return parse_canonical(f"PLACE {patterns[0]} {row} {column}")

    values = set()
    if re.search(r"\b(on|alive)\b", value):
        values.add(1)
    if re.search(r"\b(off|dead)\b", value):
        values.add(0)
    if len(values) != 1:
        raise ValueError("cell request requires exactly one alive/dead value")
    row, column = _coordinates(value, numbers)
    return parse_canonical(f"SET CELL {row} {column} {values.pop()}")


_ACTION_ATTEMPT = re.compile(
    r"\b(?:advance|step|tick|run|clear|erase|empty|put|place|add|create|remove|"
    r"turn|switch|set|make|change|toggle|glider|blinker|block)\b",
    re.I,
)


def is_read_only_conversation(text: str) -> bool:
    """Allow ordinary conversation without relaxing the mutation boundary."""
    value = " ".join(str(text).lower().strip().split())
    if not value or _ADVERSARIAL.search(value):
        return False
    if re.search(r"\b(?:toggle|random|anything|something)\b", value):
        return False
    if re.search(r"\bthen\b|;", value):
        return False
    action_attempt = bool(_ACTION_ATTEMPT.search(value))
    if action_attempt and (
        _NEGATED_OR_NEGATIVE.search(value) or _UNSUPPORTED_QUANTIFIER.search(value)
    ):
        return False
    # Valid requests such as "Can you advance three?" have already succeeded
    # through parse_user_intent before this function is consulted.
    if _QUERY_PREFIX.search(value) or value.endswith("?"):
        return True
    if action_attempt:
        return False
    return True


def resolve_implicit_placement(text: str, rng=None) -> tuple[dict, str] | None:
    """Safely complete conversational placement requests without coordinates."""
    value = " ".join(str(text).lower().strip().split())
    value = re.sub(r"\b(?:lider|glidr)\b", "glider", value)
    if not re.search(r"\b(?:put|place|add|create|drop|spawn)\b", value):
        return None
    if (
        _ADVERSARIAL.search(value)
        or _NEGATED_OR_NEGATIVE.search(value)
        or re.search(r"\bthen\b|;", value)
        or re.search(r"\b(?:advance|step|tick|run|clear|erase)\b", value)
    ):
        return None
    # Explicit or partial coordinates belong to the strict parser, which can
    # validate labels, count, and bounds without guessing.
    if _numbers(value) or re.search(r"\b(?:row|column)\b", value):
        return None
    patterns = [name for name in PATTERNS if re.search(rf"\b{name}\b", value)]
    unspecified = bool(re.search(r"\b(?:something|anything|pattern)\b", value))
    if len(patterns) > 1 or (not patterns and not unspecified):
        return None

    rng = rng or random.SystemRandom()
    random_requested = bool(re.search(r"\b(?:random|randomly|somewhere)\b", value))
    pattern = patterns[0] if patterns else (rng.choice(tuple(PATTERNS)) if random_requested else "glider")
    max_dy = max(point[0] for point in PATTERNS[pattern])
    max_dx = max(point[1] for point in PATTERNS[pattern])
    if random_requested:
        row = rng.randrange(HEIGHT - max_dy)
        column = rng.randrange(WIDTH - max_dx)
        placement = "a random valid position"
    else:
        row = (HEIGHT - (max_dy + 1)) // 2
        column = (WIDTH - (max_dx + 1)) // 2
        placement = "the center"
    command = {
        "type": "place",
        "pattern": pattern,
        "row": row,
        "column": column,
    }
    note = f"I resolved that as {canonical(command)} at {placement}"
    return command, note


def simple_fallback(text: str) -> dict:
    """Compatibility name for the fail-closed user-intent guard."""
    return parse_user_intent(text)


def canonical(command: dict) -> str:
    kind = command["type"]
    if kind == "tick":
        return f"TICK {int(command['count'])}"
    if kind == "clear":
        return "CLEAR"
    if kind == "inspect":
        return "INSPECT"
    if kind == "place":
        return (
            f"PLACE {str(command['pattern']).upper()} "
            f"{int(command['row'])} {int(command['column'])}"
        )
    if kind == "set_cell":
        return (
            f"SET CELL {int(command['row'])} {int(command['column'])} "
            f"{int(command['alive'])}"
        )
    raise ValueError(f"unsupported command: {kind!r}")


__all__ = [
    "HEIGHT",
    "PATTERNS",
    "WIDTH",
    "apply_input",
    "canonical",
    "is_read_only_conversation",
    "parse_canonical",
    "parse_user_intent",
    "resolve_implicit_placement",
    "rows_to_cells",
    "simple_fallback",
]
