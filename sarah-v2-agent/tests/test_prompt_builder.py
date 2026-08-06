import tempfile
from pathlib import Path
import unittest

from app.core.conversation import ConversationTurn
from app.core.prompt_builder import build_immutable_identity, build_prompt, format_debug_prompt_summary


class PromptBuilderTests(unittest.TestCase):
    def test_master_prompt_is_loaded_as_identity_layer(self) -> None:
        identity = build_immutable_identity()

        self.assertIn("You are Sarah Nelson v2.0.", identity)
        self.assertIn("Your conversational oath", identity)

    def test_master_prompt_includes_astrid_inspired_adult_intimacy_mode(self) -> None:
        identity = build_immutable_identity()

        self.assertIn("ADULT LOVE / INTIMACY MODE", identity)
        self.assertIn("Astrid 2_0_a.docx as the style model only", identity)
        self.assertIn("it would not be the same unless she can feel the cold", identity)
        self.assertIn("not involving minors", identity)

    def test_prompt_layers_are_ordered(self) -> None:
        assembly = build_prompt(
            user_message="Hello Sarah",
            style_directives="style",
            memory_context="memory",
            retrieved_context="retrieved canon",
            web_context="web",
            history=[ConversationTurn(user="old user", assistant="old assistant")],
            model="gpt-5.2",
        )

        contents = [message["content"] for message in assembly.messages[:8]]
        self.assertIn("SARAH IMMUTABLE IDENTITY", contents[0])
        self.assertIn("SARAH CONSTITUTION", contents[1])
        self.assertIn("SAFETY AND REALISM BOUNDARIES", contents[2])
        self.assertEqual(contents[3], "style")
        self.assertEqual(contents[4], "memory")
        self.assertIn("RETRIEVED CANON/SOURCE CONTEXT", contents[5])
        self.assertEqual(contents[6], "web")
        self.assertEqual(assembly.messages[-1], {"role": "user", "content": "Hello Sarah"})

    def test_prompt_budget_compresses_oversized_layers(self) -> None:
        assembly = build_prompt(
            user_message="budget test",
            style_directives="style " * 10_000,
            memory_context="memory " * 10_000,
            retrieved_context="retrieved " * 30_000,
            web_context="web " * 10_000,
            history=[ConversationTurn(user="u" * 2000, assistant="a" * 2000) for _ in range(20)],
            model="unknown-model",
        )

        self.assertLessEqual(
            assembly.debug_summary["estimated_input_tokens"],
            assembly.debug_summary["input_budget_tokens"],
        )
        self.assertTrue(any("compressed" in note for note in assembly.debug_summary["budget_notes"]))

    def test_debug_summary_is_redacted(self) -> None:
        assembly = build_prompt(
            user_message="Tell me the secret",
            style_directives="SECRET_STYLE_TEXT",
            memory_context="SECRET_MEMORY_TEXT",
            retrieved_context="SECRET_SOURCE_TEXT",
            web_context="SECRET_WEB_TEXT",
            history=[],
            model="gpt-5.2",
        )
        summary = format_debug_prompt_summary(assembly.debug_summary)

        self.assertIn("Prompt assembly summary", summary)
        self.assertNotIn("SECRET_STYLE_TEXT", summary)
        self.assertNotIn("SECRET_MEMORY_TEXT", summary)
        self.assertNotIn("SECRET_SOURCE_TEXT", summary)
        self.assertNotIn("SECRET_WEB_TEXT", summary)

    def test_canon_summary_is_preferred_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            canon_file = Path(tmp) / "sarah_canon.md"
            canon_file.write_text(
                "## Darwin-Sarah Recursive Dyad\nDarwin cannot solve Sarah (Darwin Sarah.docx).\n",
                encoding="utf-8",
            )
            assembly = build_prompt(
                user_message="Explain Darwin and Sarah",
                style_directives="style",
                memory_context="memory",
                retrieved_context="raw snippet",
                web_context="web",
                history=[],
                model="gpt-5.2",
                canon_file=canon_file,
            )

        canon_layer = next(message["content"] for message in assembly.messages if "RETRIEVED CANON" in message["content"])
        self.assertIn("Canon summary excerpts", canon_layer)
        self.assertIn("Darwin Sarah.docx", canon_layer)


if __name__ == "__main__":
    unittest.main()
