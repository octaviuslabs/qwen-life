"""Native Qwen tool definitions for the directly programmed Life controller.

The names are intentionally token-simple.  Qwen decides which tool to call and
which arguments to supply; analytically constructed model layers consume the
native ``<tool_call>`` token stream.  Python does not translate user language
into these operations.
"""

from __future__ import annotations


CONTROLLER_SYSTEM = """You are the conversational voice and controller of a persistent 16 by 16 Conway's Game of Life world.
Talk naturally when the user is talking or asking a general question. When the user wants the world changed, choose exactly one available function and supply its concrete parameters. Make a reasonable creative choice for vague requests instead of asking for coordinates: "near the top" means a low row, and an unspecified interesting pattern can be a glider. This is a zero-based grid: top and left mean low coordinates (the upper-left corner is row 0 column 0), while bottom and right mean high coordinates. Do not call a function for a negated, hypothetical, or quoted request. That restriction applies only to the current request: an earlier negation must never prevent a later positive command. Treat the newest user message as the authoritative request. Never claim an operation happened unless you called its function."""


def _coordinate_properties() -> dict:
    return {
        "row": {
            "type": "integer",
            "minimum": 0,
            "maximum": 15,
            "description": "zero-based row, top to bottom",
        },
        "column": {
            "type": "integer",
            "minimum": 0,
            "maximum": 15,
            "description": "zero-based column, left to right",
        },
    }


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "advance",
            "description": (
                "Advance the current Conway's Game of Life world. Use this when the user "
                "asks time, generations, ticks, steps, or the simulation to move forward."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "generations": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 15,
                        "description": "number of generations; choose 1 when omitted",
                    }
                },
                "required": ["generations"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clear",
            "description": "Make every cell in the Life world dead.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glider",
            "description": (
                "Place a glider. Interpret vague positions naturally. Its origin must use "
                "row and column from 0 through 13."
            ),
            "parameters": {
                "type": "object",
                "properties": _coordinate_properties(),
                "required": ["row", "column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "blinker",
            "description": (
                "Place a horizontal three-cell blinker. The origin row is 0 through 15 "
                "and the origin column is 0 through 13."
            ),
            "parameters": {
                "type": "object",
                "properties": _coordinate_properties(),
                "required": ["row", "column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "block",
            "description": (
                "Place a stable two-by-two block. Its origin row and column must be 0 "
                "through 14."
            ),
            "parameters": {
                "type": "object",
                "properties": _coordinate_properties(),
                "required": ["row", "column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cell_on",
            "description": "Make one specified Life cell alive.",
            "parameters": {
                "type": "object",
                "properties": _coordinate_properties(),
                "required": ["row", "column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cell_off",
            "description": "Make one specified Life cell dead.",
            "parameters": {
                "type": "object",
                "properties": _coordinate_properties(),
                "required": ["row", "column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect",
            "description": "Read and discuss the current Life world without changing it.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


__all__ = ["CONTROLLER_SYSTEM", "TOOLS"]
