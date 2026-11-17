import json
import tempfile
import unittest
from pathlib import Path

from server import (
    FileBridgeBackend,
    MockFixtureBackend,
    NaturalLanguageControllerBackend,
    Snapshot,
    command_as_natural_language,
    validate_command,
)


class RendererBridgeTests(unittest.TestCase):
    def test_snapshot_contract_rejects_nonbinary_cells(self) -> None:
        with self.assertRaises(ValueError):
            Snapshot(2, 2, (0, 1, 2, 0))

    def test_snapshot_roundtrip_preserves_runtime_event_id(self) -> None:
        value = {
            "width": 2,
            "height": 2,
            "cells": [0, 1, 1, 0],
            "generation": 3,
            "live_cells": 2,
            "event_id": 17,
        }
        assert Snapshot.from_dict(value).to_dict()["event_id"] == 17

    def test_file_bridge_forwards_command_without_changing_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state.json"
            commands = root / "commands.jsonl"
            original = {
                "width": 2,
                "height": 2,
                "cells": [1, 0, 0, 1],
                "generation": 7,
                "live_cells": 2,
            }
            state.write_text(json.dumps(original))
            backend = FileBridgeBackend(state, commands)

            before = backend.snapshot()
            result = backend.submit({"type": "tick", "count": 1})
            after = backend.snapshot()

            self.assertEqual(before.cells, after.cells)
            self.assertEqual(before.generation, after.generation)
            self.assertEqual(json.loads(state.read_text()), original)
            self.assertFalse(before.queue_pending)
            self.assertTrue(after.queue_pending)
            self.assertEqual(after.completed_offset, 0)
            self.assertEqual(result["queue_offset"], commands.stat().st_size)
            self.assertEqual(result["parsed_command"], "TICK 1")
            self.assertEqual(json.loads(commands.read_text()), {"type": "tick", "count": 1})

    def test_queue_receipts_distinguish_completion_of_two_clients(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state, commands = root / "state.json", root / "commands.jsonl"
            cursor = commands.with_suffix(".offset")
            value = {"width": 1, "height": 1, "cells": [0], "event_id": 1}
            state.write_text(json.dumps(value))
            first_client = FileBridgeBackend(state, commands)
            second_client = FileBridgeBackend(state, commands)
            first = first_client.submit({"type": "tick", "count": 1})
            second = second_client.submit({"type": "say", "text": "hello"})
            self.assertLess(first["queue_offset"], second["queue_offset"])
            self.assertEqual(second["queue_offset"], commands.stat().st_size)

            # A new state event alone cannot claim the second request finished.
            value["event_id"] = 2
            state.write_text(json.dumps(value))
            self.assertEqual(second_client.snapshot().completed_offset, 0)
            cursor.write_text(str(first["queue_offset"]))
            midway = second_client.snapshot()
            self.assertTrue(midway.queue_pending)
            self.assertEqual(midway.completed_offset, first["queue_offset"])
            self.assertLess(midway.completed_offset, second["queue_offset"])
            self.assertEqual(Snapshot.from_dict(midway.to_dict()), midway)

            value["event_id"] = 3
            state.write_text(json.dumps(value))
            cursor.write_text(str(second["queue_offset"]))
            finished = second_client.snapshot()
            self.assertFalse(finished.queue_pending)
            self.assertEqual(finished.completed_offset, second["queue_offset"])
            self.assertEqual(finished.event_id, 3)

    def test_internal_controller_translates_every_ui_mutation_to_say(self) -> None:
        cases = (
            (
                {"type": "tick", "count": 1},
                "Advance one generation.",
            ),
            ({"type": "clear"}, "Clear the entire world now."),
            (
                {"type": "place", "pattern": "glider", "row": 2, "column": 4},
                "Place a glider with its origin at row 2 column 4.",
            ),
            (
                {"type": "set_cell", "row": 7, "column": 9, "alive": 1},
                "Make the cell at row 7 column 9 alive.",
            ),
            (
                {"type": "set_cell", "row": 7, "column": 9, "alive": 0},
                "Make the cell at row 7 column 9 dead.",
            ),
        )
        for command, text in cases:
            with self.subTest(command=command):
                self.assertEqual(
                    command_as_natural_language(validate_command(command)),
                    {"type": "say", "text": text, "record_chat": False},
                )

    def test_internal_controller_queue_contains_only_say_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state.json"
            commands = root / "commands.jsonl"
            state.write_text(
                json.dumps(
                    {
                        "width": 2,
                        "height": 2,
                        "cells": [0, 0, 0, 0],
                        "generation": 0,
                        "live_cells": 0,
                    }
                )
            )
            backend = NaturalLanguageControllerBackend(FileBridgeBackend(state, commands))

            backend.submit({"type": "tick", "count": 1})
            backend.submit({"type": "clear"})
            backend.submit(
                {"type": "place", "pattern": "block", "row": 3, "column": 5}
            )
            backend.submit({"type": "set_cell", "row": 8, "column": 6, "alive": 1})
            backend.submit({"type": "say", "text": "Put something near the top."})

            queued = [json.loads(line) for line in commands.read_text().splitlines()]
            self.assertEqual(len(queued), 5)
            self.assertTrue(all(record["type"] == "say" for record in queued))
            self.assertTrue(all(record["record_chat"] is False for record in queued[:-1]))
            self.assertEqual(queued[-1], {"type": "say", "text": "Put something near the top."})

    def test_clear_chat_metadata_survives_controller_and_snapshot_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state, commands = root / "state.json", root / "commands.jsonl"
            snapshot = {
                "width": 2, "height": 2, "cells": [0, 0, 0, 0],
                "conversation": [], "conversation_epoch": 4,
            }
            state.write_text(json.dumps(snapshot))
            backend = NaturalLanguageControllerBackend(FileBridgeBackend(state, commands))

            backend.submit({"type": "clear", "reset_chat": True})

            self.assertEqual(json.loads(commands.read_text()), {
                "type": "say", "text": "Clear the entire world now.", "reset_chat": True,
                "record_chat": False,
            })
            self.assertEqual(backend.snapshot().to_dict()["conversation_epoch"], 4)
            self.assertEqual(json.loads(state.read_text()), snapshot)
            snapshot.pop("conversation_epoch")
            self.assertEqual(Snapshot.from_dict(snapshot).conversation_epoch, 0)

    def test_explicit_chat_visibility_metadata_survives_translation(self) -> None:
        for flag in (True, False):
            command = {"type": "say", "text": "hello", "record_chat": flag}
            self.assertEqual(command_as_natural_language(validate_command(command)), command)

    def test_mock_uses_recorded_frames_and_labels_itself(self) -> None:
        backend = MockFixtureBackend()
        before = backend.snapshot()
        backend.submit({"type": "tick", "count": 1})
        after = backend.snapshot()
        self.assertEqual(before.generation, 0)
        self.assertEqual(after.generation, 1)
        self.assertEqual(after.backend, "mock-recorded-frames")
        self.assertIn("no simulation", after.message)

    def test_commands_are_structural_only(self) -> None:
        self.assertEqual(validate_command({"type": "clear"}), {"type": "clear"})
        self.assertEqual(
            validate_command({"type": "set_cell", "row": 2, "column": 3, "alive": 1}),
            {"type": "set_cell", "row": 2, "column": 3, "alive": 1},
        )
        with self.assertRaises(ValueError):
            validate_command({"type": "evolve_locally"})
        with self.assertRaises(ValueError):
            validate_command(
                {"type": "place", "pattern": "glider", "row": 15, "column": 15}
            )

    def test_commands_reject_json_type_coercion_and_extra_fields(self) -> None:
        invalid = (
            {"type": "tick", "count": 1.9},
            {"type": "tick", "count": True},
            {"type": "tick", "count": "2"},
            {"type": "set_cell", "row": 1.5, "column": 2, "alive": 1},
            {"type": "set_cell", "row": 1, "column": 2, "alive": False},
            {"type": "say", "text": {"type": "clear"}},
            {"type": "place", "pattern": ["glider"], "row": 1, "column": 1},
            {"type": "clear", "count": 1},
            {"type": "clear", "reset_chat": "true"},
            {"type": "say", "text": "hello", "reset_chat": 1},
            {"type": "tick", "count": 1, "reset_chat": True},
            {"type": "say", "text": "hello", "record_chat": "false"},
            {"type": "say", "text": "hello", "record_chat": 0},
            {"type": "clear", "record_chat": False},
        )
        for command in invalid:
            with self.subTest(command=command), self.assertRaises(ValueError):
                validate_command(command)

    def test_production_sources_contain_no_life_transition_rule(self) -> None:
        root = Path(__file__).parent
        sources = (root / "server.py").read_text() + (root / "static" / "app.js").read_text()
        forbidden = ("neighbor_count", "live_neighbors", "survival", "underpopulation", "birth_rule")
        for term in forbidden:
            self.assertNotIn(term, sources)

    def test_chat_copy_and_runtime_rejection_feedback_are_safe(self) -> None:
        root = Path(__file__).parent
        page = (root / "static" / "index.html").read_text().lower()
        script = (root / "static" / "app.js").read_text()
        self.assertIn("one action at a time", page)
        self.assertNotIn("and advance twelve generations", page)
        self.assertIn("syncConversation", script)
        self.assertIn("renderedConversationIds", script)
        self.assertIn('record.kind === "error" ? "error" : "assistant"', script)


if __name__ == "__main__":
    unittest.main()
